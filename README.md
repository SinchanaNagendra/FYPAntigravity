# Blockchain-Secured Federated Learning for Multi-Bank Credit Card Fraud Detection

Final-year engineering project implementing a real, running system — no mock-ups.

Based on: Salam et al. (2024), "Privacy-Preserving Credit Card Fraud Detection"  
Extended with: secure aggregation, differential privacy, real multi-bank non-IID data,
attack evaluation, and a private Hyperledger Fabric blockchain layer.

---

## ⚠️ Stated Limitations

1. **Banks are simulated.** Neither the IEEE-CIS nor ULB dataset contains real bank
   identifiers. Banks are simulated by partitioning on pseudo-cardholder uid.
   This is a simulation of the multi-bank setting, not a deployment with truly
   distinct institutional data. See `docs/DATASET.md`.

2. **Formal DP for tree models only via k-leaf rule.** DP-SGD (Opacus) is applied to
   Logistic Regression and CNN. Decision Tree and Random Forest use `min_samples_leaf ≥ 20`
   to limit leaf-level disclosure; no formal (ε,δ)-DP guarantee is claimed.
   See `docs/DEVIATIONS.md: DEV-03`.

3. **KNN not federated.** KNN requires transmitting training data, violating the privacy
   invariant. It is only a local/centralized baseline. See `docs/DEVIATIONS.md: DEV-01`.

4. **PySyft legacy API unavailable.** PySyft 0.8+ is a data-science platform, not the FL
   toolkit from the paper. The framework comparison uses Flower (PyTorch) vs TFF instead.
   See `docs/DEVIATIONS.md: DEV-02`.

5. **The paper's claimed 99.99% accuracy is not reproduced.** That figure is almost
   certainly due to evaluation on resampled data or temporal leakage. This project uses
   a strict temporal split; results are reported as obtained.

---

## Quick Start

### Prerequisites

- Windows with WSL2 (or Linux/macOS)
- Docker + Docker Compose v2
- Python 3.10 or 3.11
- Go 1.21 (for chaincode unit tests)
- ≥ 16 GB RAM
- Kaggle account with `~/.kaggle/kaggle.json` configured

### Installation

```bash
# 1. Clone and enter workspace
cd FYPKiro

# 2. Install Python dependencies
pip install -r fl/requirements.txt

# 3. Check all dependencies
make env-check
```

### Phase 1: Data

```bash
# Download datasets (requires Kaggle API, accept ieee-fraud-detection rules first)
make data-download

# Partition into 5 banks, all scenarios
make data-partition N=5

# Run gate verification
make data-verify N=5
```

### Full Demo

```bash
# Bring up the Fabric network, run a 5-round FL experiment, open dashboard
make demo
```

---

## Project Structure

```
fabric/       Hyperledger Fabric 2.5.x network + Go chaincode
fl/           PyTorch + Flower federated learning stack
fl_tff/       TensorFlow Federated comparison (Docker)
data/         Download + partition scripts (no data committed)
attacks/      Poisoning, inference, defences
dashboard/    FastAPI backend + React frontend
experiments/  Full experiment matrix runner
results/      Generated outputs (gitignored except .gitkeep)
docs/         Architecture, Dataset, Deviations, Traceability
tests/        Unit, integration, privacy audit
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Blockchain | Hyperledger Fabric 2.5.x | Private, permissioned; MSP-based identity; no public mining |
| Ordering | Raft (3 nodes) | Production-grade BFT without PoW overhead |
| FL framework | Flower + PyTorch | Active, production-ready; Opacus-compatible |
| DP | Opacus DP-SGD | Only framework with per-sample gradient clipping for PyTorch |
| Secure aggregation | X25519 + PRG masks | Efficient, no trusted third party |
| Off-chain storage | Private IPFS (swarm.key) | Decentralised, content-addressed, non-public |
| Primary metric | PR-AUC | Robust to class imbalance (3.5% fraud) |

---

## Results Summary

All results are produced by real execution and logged in `results/`. Tables are generated in LaTeX and CSV formats in `results/tables/`, and figures in `results/figures/`.

### Centralized vs. Local vs. Federated PR-AUC ($N=5$ banks, S1 IID)

| Model | Centralized | Local (Mean) | Federated (FL) | Gap ($F - C$) | Notes |
|---|---|---|---|---|---|
| **Random Forest** | 0.3999 | 0.3945 | **0.4364** | +0.0366 | Best overall; federated ensemble grows across banks |
| **Decision Tree** | 0.3343 | 0.2846 | **0.4117** | +0.0774 | Soft-voting ensemble over 5 bank trees |
| **Logistic Regression** | 0.1330 | 0.2182 | **0.2787** | +0.1456 | Convergence across 10 rounds of FedAvg |
| **Gaussian NB** | 0.0347 | 0.0338 | 0.0317 | -0.0030 | Exact sufficient statistic aggregation |
| **KNN** | 0.2534 | 0.2297 | N/A* | — | *Non-parametric baseline; federating violates privacy (DEV-01) |

### Privacy–Utility Trade-Off (DP-SGD, Logistic Regression)

| Setting | Noise Multiplier $\sigma$ | Cumulative $\varepsilon$ ($\delta=10^{-5}$) | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| `no_dp` | 0.0 | $\infty$ | 0.2930 | 0.8195 |
| `eps_8` | 0.5 | 24.78 | 0.0812 | 0.6323 |
| `eps_3` | 0.8 | 6.12 | 0.0813 | 0.6325 |
| `eps_1` | 1.1 | 2.51 | 0.0814 | 0.6326 |
| `eps_0.5` | 1.5 | 1.41 | 0.0815 | 0.6328 |

### Adversarial Robustness & Byzantine Defences ($N=5$, Mode A vs Mode B)

- **Clean Baseline:** PR-AUC 0.2787
- **Label Flip Attack:** Recall severely collapses under 2 adversaries ($0.0886 \to 0.0142$, 84% drop in fraud detection recall).
- **Model Poisoning:** Scaled update (0.2569), Sign flip (0.2744), Backdoor (0.2770).
- **Mode B Byzantine Defences:** Trimmed mean achieves **0.3533** ($m=1$) and **0.3301** ($m=2$); Coordinate-wise median achieves **0.3024** ($m=1$) and **0.3385** ($m=2$). Krum enforces the theoretical guard ($N > 2m+2$).

### Report Assets

- **Figures:** `results/figures/fig1_privacy_utility_curve.png`, `fig2_central_vs_fed_gap.png`, `fig3_attack_results.png`, `fig4_learning_curves.png`, `fig5_balancing_comparison.png`
- **Tables:** `results/tables/tab_central_vs_fed.tex` (.csv), `tab_balancing_comparison.tex` (.csv), `tab_privacy_utility.tex` (.csv), `tab_attacks.tex` (.csv), `tab_framework_comparison.tex` (.csv)


---

## Privacy Guarantees

| Mechanism | Applies to | Formal guarantee |
|---|---|---|
| Raw data never transmitted | All models | Proven by privacy audit tests |
| Secure aggregation (X25519 + PRG) | All models | Individual updates indistinguishable from noise |
| DP-SGD (Opacus) | LR, CNN | (ε, δ)-DP per run, ε ≈ 1–8 depending on noise multiplier |
| k-leaf rule (min_samples_leaf ≥ 20) | DT, RF | Limits leaf disclosure; no formal DP |
| SHA-256 integrity | All models | Hash verified after every IPFS download |
| MSP-based identity | All functions | Only registered banks can submit |

---

## References

- Salam et al. (2024). Privacy-Preserving Credit Card Fraud Detection. [PPT source]
- Bonawitz et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. ACM CCS.
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
- Abadi et al. (2016). Deep Learning with Differential Privacy. ACM CCS.
- Vesta Corporation / IEEE-CIS Fraud Detection Competition (2019). Kaggle.
- Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling. CIDM.
