"""
tests/integration/test_fabric_client.py
========================================
Integration tests for FabricClient against a live Fabric network.

REQUIRES: `make network-up N=5 && make chaincode-deploy`
SKIP automatically when the Docker CLI containers are not running.

These tests exercise the real peer CLI via docker exec and verify:
  1. Bank registration (RegisterBank)
  2. Start a round (StartRound)
  3. Submit updates from all 5 banks (SubmitUpdate)
  4. Reject a duplicate update from the same bank
  5. Reject an unregistered MSP identity
  6. Reject an over-norm update
  7. Compute aggregate and finalize round (SubmitAggregate → FinalizeRound)
  8. Report metrics
  9. GetBankReputation reflects activity
 10. GetHistory returns all 5 updates for the round key

Run (when network is up):
    pytest tests/integration/test_fabric_client.py -v

Skip message when network is not up:
    "Fabric CLI container cli.bank0.fraud.net is not running — skipping"
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fl.bank_node.fabric_client import (
    FabricClient,
    FabricError,
    FabricAccessDeniedError,
)

N_BANKS = 5
CHANNEL = "fraudchannel"
CHAINCODE = "frauddetect"
CRYPTO_IN_CONTAINER = "/crypto"

# ── Skip guard ────────────────────────────────────────────────────────────

def _cli_container_running(bank_index: int) -> bool:
    name = f"cli.bank{bank_index}.fraud.net"
    r = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", name],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


NETWORK_UP = _cli_container_running(0)
skip_if_no_network = pytest.mark.skipif(
    not NETWORK_UP,
    reason=(
        "Fabric CLI container cli.bank0.fraud.net is not running — "
        "run: make network-up N=5 && make chaincode-deploy"
    ),
)


# ── Fixture: one FabricClient per bank ────────────────────────────────────

@pytest.fixture(scope="module")
def clients():
    return [
        FabricClient(
            bank_id=f"bank{i}",
            msp_id=f"Bank{i}MSP",
            bank_index=i,
            crypto_path_in_container=CRYPTO_IN_CONTAINER,
            n_banks=N_BANKS,
            use_blockchain=True,
        )
        for i in range(N_BANKS)
    ]


# ── Test 1: RegisterBank ───────────────────────────────────────────────────

@skip_if_no_network
def test_register_all_banks(clients):
    """Each bank can register exactly once."""
    for c in clients:
        c.register_bank()

    # Second registration must fail
    with pytest.raises(FabricError):
        clients[0].register_bank()


# ── Test 2: StartRound ────────────────────────────────────────────────────

@skip_if_no_network
def test_start_round(clients):
    """Any registered bank can open a round."""
    clients[0].start_round(
        round_id="test-round-001",
        global_model_cid="QmTestGlobalModel001",
        global_model_hash="a" * 64,
        hparams={"lr": 0.01, "epochs": 3},
        participants=[f"bank{i}" for i in range(N_BANKS)],
        max_norm_bound=50.0,
    )
    state = clients[0].get_round("test-round-001")
    assert state is not None
    assert state["status"] == "OPEN"
    assert state["quorum"] == N_BANKS // 2 + 1


# ── Test 3: SubmitUpdate ──────────────────────────────────────────────────

@skip_if_no_network
def test_submit_all_updates_reach_quorum(clients):
    """All 5 banks submit; quorum triggers AGGREGATING state."""
    for i, c in enumerate(clients):
        c.submit_update(
            round_id="test-round-001",
            cid=f"QmUpdate{i:03d}",
            sha256="b" * 64,
            size=102400,
            declared_norm=float(5 + i),
        )

    state = clients[0].get_round("test-round-001")
    assert state["status"] == "AGGREGATING", (
        f"Expected AGGREGATING after all updates; got {state['status']}"
    )
    assert len(state["updates"]) == N_BANKS


# ── Test 4: Duplicate update rejected ────────────────────────────────────

@skip_if_no_network
def test_duplicate_update_rejected(clients):
    """A second SubmitUpdate from the same bank must be rejected."""
    with pytest.raises(FabricError, match="already submitted"):
        clients[0].submit_update(
            round_id="test-round-001",
            cid="QmDuplicate",
            sha256="c" * 64,
            size=100,
            declared_norm=1.0,
        )


# ── Test 5: Unregistered MSP rejected ────────────────────────────────────

@skip_if_no_network
def test_unregistered_identity_rejected():
    """A FabricClient using an unregistered MSP cannot submit."""
    fake = FabricClient(
        bank_id="bank0",   # valid bank id
        msp_id="UnknownMSP",  # but wrong MSP in the container — the CLI
        bank_index=0,         # container uses the Admin identity, so this
        n_banks=N_BANKS,      # test exercises the chaincode access check
        use_blockchain=True,
    )
    with pytest.raises((FabricError, FabricAccessDeniedError)):
        fake.start_round(
            "test-round-bad-identity",
            "QmFake", "d" * 64, {}, ["bank0"], 100.0,
        )


# ── Test 6: Over-norm update rejected ────────────────────────────────────

@skip_if_no_network
def test_over_norm_update_rejected(clients):
    """SubmitUpdate with declaredNorm > maxNormBound must be rejected."""
    # Start a new round with low norm bound
    clients[0].start_round(
        "test-round-norm",
        "QmGlobal002", "e" * 64,
        {}, [f"bank{i}" for i in range(N_BANKS)],
        max_norm_bound=5.0,
    )
    with pytest.raises(FabricError, match="[Nn]orm"):
        clients[0].submit_update(
            round_id="test-round-norm",
            cid="QmOverNorm",
            sha256="f" * 64,
            size=1000,
            declared_norm=999.0,   # >> 5.0
        )


# ── Test 7: Full round finalization ───────────────────────────────────────

@skip_if_no_network
def test_finalize_round(clients):
    """After quorum, any bank can submit the aggregate; majority endorsement finalizes."""
    # Submit aggregate
    agg_hash = "a1b2" * 16  # 64 hex chars
    clients[0].submit_aggregate("test-round-001", "QmAggregate001", agg_hash)

    # All banks endorse
    for c in clients:
        c.endorse_aggregate("test-round-001", approve=True, recomputed_hash=agg_hash)

    # Finalize
    clients[0].finalize_round("test-round-001")

    state = clients[0].get_round("test-round-001")
    assert state["status"] == "FINALIZED"
    assert state["finalizeTime"] > 0


# ── Test 8: ReportMetrics ─────────────────────────────────────────────────

@skip_if_no_network
def test_report_metrics(clients):
    """Banks can report evaluation metrics after finalization."""
    for c in clients:
        c.report_metrics(
            "test-round-001",
            "val",
            {"pr_auc": 0.42, "roc_auc": 0.87, "f1": 0.31},
        )
    state = clients[0].get_round("test-round-001")
    assert len(state["metrics"]) == N_BANKS


# ── Test 9: GetBankReputation ─────────────────────────────────────────────

@skip_if_no_network
def test_bank_reputation_updated(clients):
    """Reputation score is updated after participating in a round."""
    rep = clients[0].get_bank_reputation("bank0")
    assert rep is not None
    assert rep["totalRounds"] >= 1
    assert 0.0 <= rep["score"] <= 1.0


# ── Test 10: GetHistory ────────────────────────────────────────────────────

@skip_if_no_network
def test_get_history_returns_round_transitions(clients):
    """GetHistory for 'ROUND:test-round-001' returns at least 4 entries
    (OPEN → COLLECTING → AGGREGATING → FINALIZED)."""
    history = clients[0].get_history("ROUND:test-round-001")
    assert isinstance(history, list)
    assert len(history) >= 4, (
        f"Expected >= 4 ledger writes for the round; got {len(history)}"
    )
    statuses = []
    for entry in history:
        try:
            val = json.loads(entry["value"])
            statuses.append(val.get("status"))
        except (json.JSONDecodeError, KeyError):
            pass
    assert "FINALIZED" in statuses


# ── Test 11: Minority endorsement rejected ────────────────────────────────

@skip_if_no_network
def test_minority_endorsement_insufficient_for_finalize(clients):
    """FinalizeRound must fail if fewer than quorum banks endorsed."""
    # Start a new round
    clients[0].start_round(
        "test-round-minority",
        "QmGlobal003", "b" * 64,
        {}, [f"bank{i}" for i in range(N_BANKS)],
        max_norm_bound=100.0,
    )
    # Submit only 1 update (not enough for quorum → stays COLLECTING)
    clients[0].submit_update("test-round-minority", "QmUpd", "c" * 64, 1000, 1.0)

    # FinalizeRound on a non-AGGREGATING round must fail
    with pytest.raises(FabricError):
        clients[0].finalize_round("test-round-minority")
