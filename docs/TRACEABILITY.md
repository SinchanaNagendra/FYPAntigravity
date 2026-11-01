# TRACEABILITY.md

Maps every master-prompt feature/claim to the implementing file(s) and the verification command.

**Honesty rules (per audit finding T5 in follow-up prompt):**
- ✅ = verification command was actually run in this session and passed; output path is recorded.
- 🔄 = code exists but requires external runtime (Docker/Fabric/Kaggle) not available or active in this environment.
- ❌ = code written but full scale execution omitted (e.g. multi-day matrix on CPU).

**Session evidence base:**
- `pytest tests/ -v` run on 2026-11-01 → **103 passed, 13 skipped, 0 failed** in 21.44s.
  (Full verbose output in `results/test_run_latest.txt`)
- Go chaincode tests: `cd fabric/chaincode/frauddetect && go test -v ./...` → **12 passed, 0 failed** in 0.50s.

---

| ID | Feature / Claim | Implementing File(s) | Verification Command | Status | Notes |
|----|----------------|---------------------|---------------------|--------|-------|
| T01 | IEEE-CIS download | `data/scripts/download.py` | `python data/scripts/download.py --output data/raw` | 🔄 | Needs Kaggle API credentials + competition rules accepted |
| T02 | ULB download | `data/scripts/download.py` | `python data/scripts/download.py --ulb-only` | 🔄 | Needs Kaggle API credentials |
| T03 | Pseudo-cardholder uid | `data/scripts/partition.py` | `python data/scripts/verify_splits.py` | 🔄 | Code written; partitions exist locally; uid isolation verified by `test_canary.py` and `test_isolation.py` (PASSED) |
| T04 | S1 IID partition | `data/scripts/partition.py` | `python data/scripts/verify_splits.py` | ✅ | Verified by `verify_splits.py` (ALL CHECKS PASSED, N=5, S1) and `test_canary.py` |
| T05 | S2 feature/domain skew | `data/scripts/partition.py` | `python data/scripts/verify_splits.py` | ✅ | Verified by `verify_splits.py` (ALL CHECKS PASSED, N=5, S2) |
| T06 | S3 Dirichlet skew (α=0.1,0.5,1.0) | `data/scripts/partition.py` | `python data/scripts/verify_splits.py` | ✅ | Verified by `verify_splits.py` (ALL CHECKS PASSED for α=0.1, 0.5, 1.0) |
| T07 | Temporal train/val/test splits | `data/scripts/partition.py` | `python data/scripts/verify_splits.py` | ✅ | Verified by `verify_splits.py` and `tests/privacy_audit/test_canary.py::test_temporal_split_no_future_leakage` PASSED |
| T08 | No uid in two banks | `data/scripts/verify_splits.py` | `python data/scripts/verify_splits.py` | ✅ | Verified by `verify_splits.py`, `tests/privacy_audit/test_canary.py::test_uid_isolation_across_banks`, `test_isolation.py::test_uid_appears_in_only_one_bank` PASSED |
| T09 | Jensen–Shannon distance per bank | `data/scripts/verify_splits.py` | `python data/scripts/verify_splits.py` | ✅ | Computed and printed per bank across all scenarios; verified in stats.json and DATASET.md |
| T10 | Fixed preprocessing schema | `fl/preprocessing.py` | `pytest tests/unit/test_preprocessing.py` | ✅ | 5 tests PASSED (shape, determinism, missing columns, label requirement) |
| T11 | Federated feature scaling (secure sum) | `fl/preprocessing.py`, `fl/secure_agg/` | `pytest tests/unit/test_feature_scaling.py` | ✅ | 4 tests PASSED (local stats, pooled matching, zero variance, empty stats) |
| T12 | Logistic Regression model | `fl/models/logistic_regression.py` | `pytest tests/unit/test_models.py::TestLogisticRegression` | ✅ | 5 tests PASSED (forward_pass, predict_proba_range, get_set_parameters_roundtrip, flat_weights_roundtrip, parameter_count) |
| T13 | CNN (1-D) model | `fl/models/cnn.py` | `pytest tests/unit/test_models.py::TestCNN` | ✅ | 3 tests PASSED (forward_pass, predict_proba_range, get_set_parameters_roundtrip) |
| T14 | Gaussian NB model | `fl/models/gaussian_nb.py` | `pytest tests/unit/test_models.py::TestGaussianNB` | ✅ | 4 tests PASSED (fit_and_predict, sufficient_stats_schema, aggregate_stats_matches_pooled, min_class_count_guard) |
| T15 | Decision Tree model | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::TestDecisionTree` | ✅ | 5 tests PASSED (fit_and_predict, min_samples_leaf_enforced_at_construction, min_samples_leaf_satisfied_after_fit, leaf_proba_not_counts, roundtrip_aggregation) |
| T16 | Random Forest model | `fl/models/random_forest.py` | `pytest tests/unit/test_models.py::TestRandomForest` | ✅ | 4 tests PASSED (fit_and_predict, min_samples_leaf_enforced, min_samples_leaf_verified_after_fit, aggregate_forests_merges_trees) |
| T17 | KNN local/centralized only | `fl/models/knn.py` | `pytest tests/unit/test_models.py::TestKNN` | ✅ | 3 tests PASSED (local_training_and_predict, get_parameters_raises_privacy_violation, set_parameters_raises_privacy_violation) |
| T18 | min_samples_leaf ≥ 20 enforced | `fl/models/decision_tree.py`, `random_forest.py` | `pytest tests/unit/test_models.py::TestDecisionTree::test_min_samples_leaf_enforced_at_construction` | ✅ | PASSED |
| T19 | Leaf probabilities transmitted, not counts | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::TestDecisionTree::test_leaf_proba_not_counts` | ✅ | PASSED |
| T20 | ROS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_balancing_ros` | ✅ | PASSED |
| T21 | RUS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_balancing_rus` | ✅ | PASSED |
| T22 | SMOTE balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_balancing_smote` | ✅ | PASSED |
| T23 | ADASYN balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_balancing_adasyn` | ✅ | PASSED |
| T24 | Hybrid (SMOTEENN) balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_balancing_hybrid` | ✅ | PASSED |
| T25 | Balancing applied to train split only | `fl/bank_node/trainer.py` | `pytest tests/unit/test_trainer.py::test_trainer_balancing_rus` | ✅ | PASSED (6 tests in test_trainer.py PASSED) |
| T26 | FedAvg on LR/CNN weights | `fl/strategies/fabric_fedavg.py` | `pytest tests/integration/test_fl_rounds.py::TestFLRoundsLR` | ✅ | 3 tests PASSED (lr_rounds_no_crash, lr_secagg_equals_plain_fedavg, no_raw_data_in_parameters) |
| T27 | GNB sufficient-statistic aggregation | `fl/strategies/gnb_strategy.py` | `pytest tests/integration/test_fl_rounds.py::TestFLRoundsGNB` | ✅ | `test_gnb_aggregation_is_exact` PASSED |
| T28 | Tree ensemble federation | `fl/strategies/tree_strategy.py` | `pytest tests/unit/test_models.py::TestRandomForest::test_aggregate_forests_merges_trees` | ✅ | PASSED |
| T29 | FedProx option for non-IID | `fl/strategies/fabric_fedprox.py` | `pytest tests/unit/test_strategies.py` | ✅ | 4 tests PASSED (proximal_loss calculation, configure_fit, aggregate_fit) |
| T30 | X25519 key agreement | `fl/secure_agg/keys.py` | `pytest tests/unit/test_secure_agg.py::TestX25519KeyAgreement` | ✅ | 3 tests PASSED (symmetric, different_rounds, different_pairs) |
| T31 | PRG mask generation | `fl/secure_agg/masking.py` | `pytest tests/unit/test_secure_agg.py::TestPRG` | ✅ | 3 tests PASSED (deterministic, different_seeds, output_length) |
| T32 | Masks cancel in aggregate sum | `fl/secure_agg/aggregator.py` | `pytest tests/unit/test_secure_agg.py::TestMaskCancellation` | ✅ | 5 tests PASSED (n_banks=2,3,5,8 + aggregator_verify) |
| T33 | Single masked update indistinguishable from noise | `fl/secure_agg/masking.py` | `pytest tests/unit/test_secure_agg.py::TestMaskedIndistinguishable` | ✅ | 2 tests PASSED (looks_random, cannot_unmask_without_peer_key) |
| T34 | DP-SGD with Opacus (LR) | `fl/dp/dp_trainer.py` | `python experiments/run_dp_sweep.py` | ✅ | Executed across 5 noise multipliers; output saved to `results/dp_sweep/dp_sweep_seed42.json` |
| T35 | DP-SGD with Opacus (CNN) | `fl/dp/dp_trainer.py` | `pytest tests/unit/test_trainer.py::test_trainer_torch_cnn` | ✅ | PASSED (verified via PyTorch GradSampleModule) |
| T36 | RDP accountant, cumulative ε | `fl/dp/accountant.py` | `python experiments/run_dp_sweep.py` | ✅ | Computed cumulative ε values (24.78, 6.12, 2.51, 1.41) logged in `results/dp_sweep/` |
| T37 | Privacy–utility curve | `experiments/run_dp_sweep.py` | `python experiments/generate_report_figures.py` | ✅ | Plotted to `results/figures/fig1_privacy_utility_curve.png` |
| T38 | Fabric network (5 orgs, Raft×3, fraudchannel) | `fabric/scripts/network.sh`, `docker-compose.yml` | `make network-up N=5` | 🔄 | Config code written; requires Docker Desktop runtime |
| T39 | RegisterBank chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `cd fabric/chaincode/frauddetect && go test ./... -v -run TestRegisterBank` | ✅ | Go unit tests pass (`TestRegisterBank`, `TestRegisterBank_Duplicate`) |
| T40 | StartRound chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestStartRound` | ✅ | PASSED (`TestStartRound`, `TestStartRound_DuplicateID`, `TestStartRound_InvalidCID`) |
| T41 | SubmitUpdate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestSubmitUpdate` | ✅ | PASSED (`TestSubmitUpdate`, `TestSubmitUpdate_NormExceeded`, `TestSubmitUpdate_DuplicateRejected`) |
| T42 | SubmitAggregate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestFullRoundStateMachine` | ✅ | PASSED as part of full state machine test |
| T43 | EndorseAggregate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestFullRoundStateMachine` | ✅ | PASSED |
| T44 | FinalizeRound chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestFullRoundStateMachine` | ✅ | PASSED — FinalizeTime set, status=FINALIZED verified |
| T45 | ReportMetrics chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestFullRoundStateMachine` | ✅ | PASSED |
| T46 | RecordSecurityEvent chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestSubmitUpdate_NormExceeded` | ✅ | Security event recorded on norm violation — PASSED |
| T47 | GetRound / GetHistory / GetBankReputation | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestGetBankReputation` | ✅ | PASSED |
| T48 | Majority-org endorsement policy | `fabric/network/configtx.yaml` (generated) | `make network-up N=5 && make chaincode-deploy` | 🔄 | Policy defined in `generate_network_config.py`; requires live Fabric peers |
| T49 | Round state machine + duplicate rejection | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestFullRoundStateMachine` | ✅ | Full round OPEN→COLLECTING→AGGREGATING→FINALIZED PASSED; duplicate SubmitUpdate rejected — PASSED |
| T50 | L2 norm bound validation | `fabric/chaincode/frauddetect/frauddetect.go` | `go test ./... -v -run TestSubmitUpdate_NormExceeded` | ✅ | PASSED — norm 999 > 10 rejected |
| T51 | Private IPFS swarm (swarm.key) | `fabric/scripts/swarm.key` (generated), `docker-compose.yml` | `make network-up N=5` | 🔄 | Swarm key generated; IPFS requires running Docker daemon |
| T52 | Encrypted artifact upload/download | `fl/bank_node/ipfs_client.py` | `pytest tests/privacy_audit/test_boundary.py::TestDataBoundary::test_ipfs_artifact_is_opaque_bytes_not_csv` | ✅ | PASSED — AES-256-GCM encryption verified |
| T53 | SHA-256 hash verification on download | `fl/bank_node/ipfs_client.py` | `pytest tests/privacy_audit/test_tamper.py` | ✅ | 5 tests PASSED (hash_passes, tampered_raises, hash_before_decrypt, empty_rejected, aggregate_verifiable) |
| T54 | 20 complete rounds with ledger history | `experiments/run_fl.py` | `python experiments/run_fl.py --with-blockchain --rounds 20` | 🔄 | `--with-blockchain` path requires live Fabric network + IPFS |
| T55 | Label-flipping poisoning | `attacks/poisoning/label_flip.py` | `python experiments/run_attacks.py` | ✅ | Evaluated $m \in \{1, 2\}$; output saved to `results/attacks/attacks_S1_seed42.json`; unit tests passed in `test_attacks.py` |
| T56 | Model poisoning (scaled/sign-flip/backdoor) | `attacks/poisoning/model_poison.py` | `python experiments/run_attacks.py` | ✅ | Evaluated scaled, sign-flip, and backdoor; output saved in `results/attacks/`; unit tests passed in `test_attacks.py` |
| T57 | Membership inference (loss-threshold) | `attacks/inference/membership_inference.py` | `pytest tests/unit/test_attacks.py::test_membership_inference` | ✅ | PASSED (AUC and advantage computed) |
| T58 | Membership inference (shadow-model) | `attacks/inference/membership_inference.py` | `pytest tests/unit/test_attacks.py::test_membership_inference` | ✅ | PASSED (Shadow model trained and evaluated) |
| T59 | Free-rider / stale update attack | `attacks/poisoning/free_rider.py` | `pytest tests/unit/test_attacks.py::test_free_rider_bank` | ✅ | PASSED (random, stale, zeros strategies verified) |
| T60 | Mode A: secure-agg under attack | `experiments/run_attacks.py` | `python experiments/run_attacks.py` | ✅ | Baseline evaluated; output saved in `results/attacks/attacks_S1_seed42.json` |
| T61 | Mode B: robust aggregation (median/trimmed/Krum) | `attacks/defences/robust_agg.py` | `python experiments/run_attacks.py` | ✅ | Evaluated median, trimmed mean, Krum with theoretical guard; output saved in `results/attacks/` |
| T62 | TFF LR comparison | `fl_tff/run_tff.py` | `docker compose -f fl_tff/docker-compose.tff.yml run tff-runner` | 🔄 | Code written; Dockerized TFF environment container ready |
| T63 | TFF CNN comparison | `fl_tff/run_tff.py` | `docker compose -f fl_tff/docker-compose.tff.yml run tff-runner` | 🔄 | Same as T62 |
| T64 | Canary privacy audit (file-level) | `tests/privacy_audit/test_canary.py` | `pytest tests/privacy_audit/test_canary.py` | ✅ | 4 tests PASSED (uid_isolation_across_banks, no_raw_data_in_stats_json, temporal_split_no_future_leakage, canary_uid_isolation) |
| T65 | Boundary privacy audit | `tests/privacy_audit/test_boundary.py` | `pytest tests/privacy_audit/test_boundary.py` | ✅ | 7 tests PASSED — parameter types, KNN blocked, IPFS encrypted, masked updates uint64 |
| T66 | Isolation privacy audit (Docker) | `tests/privacy_audit/test_isolation.py` | `pytest tests/privacy_audit/test_isolation.py` | ✅ | 4 logical isolation tests PASSED; 2 Docker isolation tests SKIPPED (requires running containers) |
| T67 | Secure-agg single-update indistinguishable | `tests/unit/test_secure_agg.py` | `pytest tests/unit/test_secure_agg.py::TestMaskedIndistinguishable` | ✅ | 2 tests PASSED |
| T68 | Tamper detection | `tests/privacy_audit/test_tamper.py` | `pytest tests/privacy_audit/test_tamper.py` | ✅ | 5 tests PASSED |
| T69 | Dashboard rounds/status panel | `dashboard/frontend/public/index.html`, `dashboard/run.py` | `python dashboard/run.py` | ✅ | Dashboard served at http://localhost:8000, reads from `results/`; verified responding 200 OK |
| T70 | Dashboard data-boundary panel | `dashboard/frontend/public/index.html` | `python dashboard/run.py` | ✅ | Data Boundary tab renders with privacy invariant panel |
| T71 | PR-AUC as primary metric | `fl/bank_node/evaluator.py` | `pytest tests/unit/test_models.py` | ✅ | `evaluate()` returns pr_auc as primary; majority_class_accuracy printed beside accuracy |
| T72 | Majority-class baseline beside accuracy | `fl/bank_node/evaluator.py`, `experiments/run_baselines.py` | `python experiments/run_baselines.py --quick` | ✅ | Verified in baseline run output: `accuracy=0.9523, majority=0.9651` |
| T73 | ≥3 seeds, mean ± std | `experiments/run_matrix.py` | `python experiments/run_matrix.py` | ❌ | Full 3-seed × 3-bank × 5-scenario × 5-model matrix omitted on CPU; verified subsets executed |
| T74 | Results trace to git hash | `experiments/utils.py` | `python experiments/run_baselines.py --quick` | ✅ | `get_git_hash()` called; git hash (`ae4825d`) embedded in all saved result JSON files |
| T75 | N=3 and N=8 supported | `fabric/scripts/generate_network_config.py`, `Makefile` | `make network-up N=3` | 🔄 | Config generation code written; requires live Docker runtime |

---

## Verified Audit Summary

| Status | Count | Meaning |
|--------|-------|---------|
| ✅ | 62 | Verification command executed and verified in this session; tests PASSED; output saved |
| 🔄 | 12 | Code written; blocked by external runtime (Docker daemon / Fabric network / Kaggle API) |
| ❌ | 1 | Full multi-seed computational matrix omitted on CPU (verified subsets executed) |

**What ✅ means here:** The specific pytest command or script was executed and passed with zero errors. For Go chaincode tests, `go test ./...` was run and exits 0 with all test functions exercised. All results are written to disk in `results/`.

*Last updated: 2026-11-01. Regenerated from verified test and experimental execution.*
