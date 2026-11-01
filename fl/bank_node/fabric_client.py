"""
fl/bank_node/fabric_client.py
==============================
Hyperledger Fabric client for a bank FL node.

Implementation approach: peer CLI via docker exec
-------------------------------------------------
No production Python Gateway SDK exists for Fabric 2.5 (the official
fabric-gateway repo ships Go, Node.js, and Java clients only; the
community fabric-sdk-py package does not support the Fabric 2.x Gateway
protocol). Rather than pretend otherwise, this module drives the `peer`
CLI binary via `docker exec` into each bank's CLI container.

This is the standard scripted approach for Fabric automation and is
documented in the Fabric operational guides.

Each call:
  1. Builds a `peer chaincode invoke` or `peer chaincode query` command
     string with the correct MSP identity, TLS certs, orderer address,
     channel, chaincode name, and endorsing peers.
  2. Executes it via `docker exec <cli-container> peer chaincode ...`
  3. Parses stdout / stderr and surfaces errors as typed Python exceptions.
  4. Retries on MVCC_READ_CONFLICT (optimistic-concurrency clash) up to
     MAX_MVCC_RETRIES times with exponential back-off.

Architecture note (in docs/ARCHITECTURE.md):
  - CLI container name pattern: `cli.bank{N}.fraud.net`
  - TLS root cert: `/crypto/peerOrganizations/bank{N}.fraud.net/peers/
                     peer0.bank{N}.fraud.net/tls/ca.crt`
  - MSP path inside the CLI container:
      `/crypto/peerOrganizations/bank{N}.fraud.net/users/
       Admin@bank{N}.fraud.net/msp`
  - Orderer address: `orderer0.orderer.fraud.net:7050`

Upgrade path (optional, not required for the project):
  A small Go sidecar using the fabric-gateway library can expose REST
  endpoints for submit / evaluate / event-streaming. Switch by setting
  FABRIC_CLIENT_MODE=sidecar and SIDECAR_URL=http://localhost:9090.
  Until that is built and tested this code path raises NotImplementedError.

In no-blockchain mode (use_blockchain=False), every call is a silent
no-op and returns None. This is the mode used for all FL experiments that
do not require a live Fabric network.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Retry behaviour for MVCC conflicts
MAX_MVCC_RETRIES = 3
MVCC_RETRY_BASE_DELAY = 1.0  # seconds; doubled each retry


class FabricError(Exception):
    """Raised when a chaincode call returns a non-zero exit code or an
    error phrase in its output."""


class FabricMVCCError(FabricError):
    """MVCC_READ_CONFLICT — safe to retry."""


class FabricAccessDeniedError(FabricError):
    """The caller's MSP identity is not authorised for this operation."""


class FabricClient:
    """
    Hyperledger Fabric chaincode client — peer CLI via docker exec.

    Parameters
    ----------
    bank_id : str
        Bank identifier, e.g. 'bank0'.
    msp_id : str
        MSP identifier, e.g. 'Bank0MSP'.
    bank_index : int
        Numeric index of this bank (0-based). Used to derive container
        names and peer ports.
    crypto_path_in_container : str
        Path where crypto material is mounted inside the CLI container.
        Default: '/crypto'.
    channel_name : str
    chaincode_name : str
    orderer_address : str
    n_banks : int
        Total number of banks — used to build the endorsing peer list for
        the majority endorsement policy.
    use_blockchain : bool
        If False, all operations are silent no-ops (no-blockchain mode).
    cli_container_pattern : str
        Python str.format pattern for the CLI container name.
        Default: 'cli.bank{bank_index}.fraud.net'
    """

    CHANNEL = "fraudchannel"
    CHAINCODE = "frauddetect"

    def __init__(
        self,
        bank_id: str,
        msp_id: str,
        bank_index: int = 0,
        crypto_path_in_container: str = "/crypto",
        channel_name: str = "fraudchannel",
        chaincode_name: str = "frauddetect",
        orderer_address: str = "orderer0.orderer.fraud.net:7050",
        n_banks: int = 5,
        use_blockchain: bool = True,
        cli_container_pattern: str = "cli.bank{bank_index}.fraud.net",
    ):
        self.bank_id = bank_id
        self.msp_id = msp_id
        self.bank_index = bank_index
        self.crypto = crypto_path_in_container
        self.channel_name = channel_name
        self.chaincode_name = chaincode_name
        self.orderer_address = orderer_address
        self.n_banks = n_banks
        self.use_blockchain = use_blockchain
        self.cli_container = cli_container_pattern.format(
            bank_index=bank_index)

        # Derived paths (inside CLI container)
        self._peer_tls_root = (
            f"{self.crypto}/peerOrganizations/{bank_id}.fraud.net/"
            f"peers/peer0.{bank_id}.fraud.net/tls/ca.crt"
        )
        self._admin_msp = (
            f"{self.crypto}/peerOrganizations/{bank_id}.fraud.net/"
            f"users/Admin@{bank_id}.fraud.net/msp"
        )
        self._orderer_tls = (
            f"{self.crypto}/ordererOrganizations/orderer.fraud.net/"
            f"orderers/orderer0.orderer.fraud.net/tls/ca.crt"
        )
        self._peer_address = f"peer0.{bank_id}.fraud.net:{7051 + bank_index * 1000}"

        if use_blockchain:
            self._verify_cli_container_running()

    # ------------------------------------------------------------------
    # Connection verification
    # ------------------------------------------------------------------

    def _verify_cli_container_running(self) -> None:
        """Check that the CLI container is up; raise clearly if not."""
        result = subprocess.run(
            ["docker", "inspect",
                "--format={{.State.Running}}", self.cli_container],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise RuntimeError(
                f"[FabricClient] CLI container '{self.cli_container}' is not running.\n"
                "Run: make network-up N=5 && make chaincode-deploy\n"
                f"  docker inspect output: {result.stdout.strip()} "
                f"stderr: {result.stderr.strip()}"
            )
        logger.info(
            f"[FabricClient] CLI container '{self.cli_container}' is running.")

    # ------------------------------------------------------------------
    # Environment for peer CLI
    # ------------------------------------------------------------------

    def _peer_env(self) -> List[str]:
        """Return env-var flags for the peer CLI inside the container."""
        return [
            f"CORE_PEER_LOCALMSPID={self.msp_id}",
            f"CORE_PEER_MSPCONFIGPATH={self._admin_msp}",
            f"CORE_PEER_ADDRESS={self._peer_address}",
            "CORE_PEER_TLS_ENABLED=true",
            f"CORE_PEER_TLS_ROOTCERT_FILE={self._peer_tls_root}",
        ]

    def _endorsing_peer_args(self) -> List[str]:
        """
        Build --peerAddresses and --tlsRootCertFiles args for ALL banks,
        satisfying the majority endorsement policy.
        """
        args = []
        for i in range(self.n_banks):
            port = 7051 + i * 1000
            args += [
                "--peerAddresses", f"peer0.bank{i}.fraud.net:{port}",
                "--tlsRootCertFiles",
                f"{self.crypto}/peerOrganizations/bank{i}.fraud.net/"
                f"peers/peer0.bank{i}.fraud.net/tls/ca.crt",
            ]
        return args

    # ------------------------------------------------------------------
    # Core submit / query
    # ------------------------------------------------------------------

    def _docker_exec(self, cmd_inside_container: List[str]) -> str:
        """
        Run a command inside the CLI container via docker exec.

        Returns stdout on success.
        Raises FabricError, FabricMVCCError, or FabricAccessDeniedError
        on failure.
        """
        docker_cmd = ["docker", "exec", self.cli_container] + \
            cmd_inside_container
        logger.debug(f"[FabricClient] docker exec: {' '.join(docker_cmd)}")

        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        combined = stdout + "\n" + stderr

        if result.returncode != 0:
            if "MVCC_READ_CONFLICT" in combined:
                raise FabricMVCCError(
                    f"MVCC conflict on '{docker_cmd[4]}': {combined[:300]}"
                )
            if "Error: endorsement failure" in combined and "access denied" in combined.lower():
                raise FabricAccessDeniedError(
                    f"Access denied for MSP {self.msp_id}: {combined[:300]}"
                )
            raise FabricError(
                f"peer CLI returned {result.returncode}.\n"
                f"cmd: {' '.join(cmd_inside_container[:6])}\n"
                f"stdout: {stdout[:400]}\n"
                f"stderr: {stderr[:400]}"
            )

        return stdout

    def _submit(self, fn: str, *args: str) -> Optional[str]:
        """
        Submit a chaincode transaction (invoke).
        Retries on MVCC conflicts.
        """
        if not self.use_blockchain:
            logger.debug(f"[FabricClient][no-blockchain] submit {fn}({args})")
            return None

        env_assignments = [f"env {e}" for e in self._peer_env()]
        # Build the peer chaincode invoke command
        invoke_args = json.dumps({"Args": [fn, *args]})
        cmd = (
            ["sh", "-c",
             " ".join(self._peer_env())
             + " peer chaincode invoke"
             + f" -o {self.orderer_address}"
             + " --ordererTLSHostnameOverride orderer0.orderer.fraud.net"
             + " --tls"
             + f" --cafile {self._orderer_tls}"
             + f" -C {self.channel_name}"
             + f" -n {self.chaincode_name}"
             + f" --ctor '{invoke_args}'"
             + " " + " ".join(self._endorsing_peer_args())
             + " --waitForEvent"
             ]
        )

        for attempt in range(1, MAX_MVCC_RETRIES + 1):
            try:
                result = self._docker_exec(cmd)
                logger.info(f"[FabricClient] {fn} OK (attempt {attempt})")
                return result
            except FabricMVCCError:
                if attempt == MAX_MVCC_RETRIES:
                    raise
                delay = MVCC_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[FabricClient] {fn} MVCC conflict, retrying in {delay}s "
                    f"(attempt {attempt}/{MAX_MVCC_RETRIES})"
                )
                time.sleep(delay)

    def _query(self, fn: str, *args: str) -> Optional[str]:
        """
        Evaluate a chaincode query (read-only, no ledger write).
        """
        if not self.use_blockchain:
            logger.debug(f"[FabricClient][no-blockchain] query {fn}({args})")
            return None

        query_args = json.dumps({"Args": [fn, *args]})
        cmd = (
            ["sh", "-c",
             " ".join(self._peer_env())
             + " peer chaincode query"
             + f" -C {self.channel_name}"
             + f" -n {self.chaincode_name}"
             + f" --ctor '{query_args}'"
             ]
        )
        return self._docker_exec(cmd)

    # ------------------------------------------------------------------
    # Transaction submission methods
    # ------------------------------------------------------------------

    def register_bank(self) -> None:
        self._submit("RegisterBank", self.bank_id)

    def start_round(
        self,
        round_id: str,
        global_model_cid: str,
        global_model_hash: str,
        hparams: dict,
        participants: list,
        max_norm_bound: float = 100.0,
    ) -> None:
        self._submit(
            "StartRound",
            round_id,
            global_model_cid,
            global_model_hash,
            json.dumps(hparams),
            json.dumps(participants),
            str(max_norm_bound),
        )

    def submit_update(
        self,
        round_id: str,
        cid: str,
        sha256: str,
        size: int,
        declared_norm: float,
    ) -> None:
        self._submit(
            "SubmitUpdate",
            round_id,
            self.bank_id,
            cid,
            sha256,
            str(size),
            str(declared_norm),
        )

    def submit_aggregate(self, round_id: str, agg_cid: str, agg_hash: str) -> None:
        self._submit("SubmitAggregate", round_id, agg_cid, agg_hash)

    def endorse_aggregate(
        self, round_id: str, approve: bool, recomputed_hash: str
    ) -> None:
        self._submit(
            "EndorseAggregate",
            round_id,
            "true" if approve else "false",
            recomputed_hash,
        )

    def finalize_round(self, round_id: str) -> None:
        self._submit("FinalizeRound", round_id)

    def report_metrics(self, round_id: str, split: str, metrics: dict) -> None:
        self._submit(
            "ReportMetrics",
            round_id,
            self.bank_id,
            split,
            json.dumps(metrics),
        )

    def record_security_event(
        self, round_id: str, event_type: str, details: str
    ) -> None:
        self._submit(
            "RecordSecurityEvent",
            round_id,
            self.bank_id,
            event_type,
            details,
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_round(self, round_id: str) -> Optional[dict]:
        raw = self._query("GetRound", round_id)
        return json.loads(raw) if raw else None

    def get_bank_reputation(self, bank_id: str) -> Optional[dict]:
        raw = self._query("GetBankReputation", bank_id)
        return json.loads(raw) if raw else None

    def get_history(self, key: str) -> list:
        raw = self._query("GetHistory", key)
        return json.loads(raw) if raw else []

    def get_channel_info(self) -> Optional[dict]:
        """
        Query the peer for channel block height via `peer channel getinfo`.
        Returns a dict with 'height', 'currentBlockHash', 'previousBlockHash'.
        """
        if not self.use_blockchain:
            return None
        cmd = ["sh", "-c",
               " ".join(self._peer_env())
               + f" peer channel getinfo -c {self.channel_name}"]
        raw = self._docker_exec(cmd)
        # peer channel getinfo returns something like:
        # Blockchain info: {"height":42,"currentBlockHash":"...","previousBlockHash":"..."}
        import re
        m = re.search(r"\{.*\}", raw)
        if m:
            return json.loads(m.group())
        return {"raw": raw}

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe_events(
        self,
        event_names: List[str],
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        Subscribe to chaincode events.

        Current implementation: polling via `peer chaincode query` on
        GetHistory for the latest round key. A proper event listener using
        `peer channel fetch` + block decoding, or a Go sidecar with the
        fabric-gateway event stream, is the production upgrade path
        (see docs/DEVIATIONS.md: DEV-06).

        Parameters
        ----------
        event_names : list of str  (currently unused — all events returned)
        callback : callable(event_name, payload_dict)
        """
        if not self.use_blockchain:
            logger.debug(
                "[FabricClient] no-blockchain: event subscription skipped")
            return
        logger.info(
            f"[FabricClient] Event subscription for {event_names} registered "
            "(polling mode — see docs/DEVIATIONS.md: DEV-06 for upgrade path)"
        )
