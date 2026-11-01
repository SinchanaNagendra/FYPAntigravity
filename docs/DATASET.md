# DATASET.md

## Data Sources

### Primary: IEEE-CIS Fraud Detection (Vesta Corporation)

- **Source:** Kaggle competition `ieee-fraud-detection`
- **URL:** https://www.kaggle.com/competitions/ieee-fraud-detection
- **Description:** Real e-commerce card transaction data from Vesta Corporation.
  590,540 labelled transactions in the training set. ~3.5% fraud rate.
- **Features:** TransactionDT, TransactionAmt, ProductCD, card1–card6, address fields,
  email domains, C* (count features), D* (time delta features), M* (match features),
  V1–V339 (Vesta-engineered features), identity/device features from linked identity table.
- **Join:** `train_transaction` + `train_identity` on `TransactionID` (left join).
- **Academic use:** Permitted for non-commercial research per Kaggle competition rules.
  Cite as: Vesta Corporation / IEEE Computational Intelligence Society Fraud Detection (2019).
  Competition organizers: IEEE-CIS.
- **Download:** `python data/scripts/download.py`

### Secondary: ULB Credit Card Fraud Detection

- **Source:** Kaggle dataset `mlg-ulb/creditcardfraud`
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Description:** 284,807 European credit card transactions, 492 fraudulent (~0.17%).
  Features: Time, Amount, V1–V28 (PCA-anonymised), Class (0/1).
- **Usage:** Paper comparison tables ONLY (same algorithms, same balancing methods,
  under both IID and non-IID splits, following the Salam et al. (2024) setup).
  Not used for the main federated experiments.
- **Privacy:** PCA anonymisation preserves privacy of individual cardholders.
- **Citation:** Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling for
  Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM).

---

## IMPORTANT: Simulated Multi-Bank Setting

**Neither dataset contains a real bank identifier.**

Banks are simulated by partitioning a single real-world dataset using a
pseudo-cardholder key. This is an explicit simulation of the multi-bank
setting, not a deployment with truly distinct institutional data.
**This limitation is stated in all experiment tables, the README, and the report.**

### Pseudo-Cardholder Key Construction

```
uid = SHA-256(str(card1) || "|" || str(card2) || "|" || str(card3) || "|"
            || str(card4) || "|" || str(card5) || "|" || str(addr1))[:16]
```

- `card1`–`card5` are masked card identifiers; `addr1` is a billing address bucket.
- Using the first 16 hex characters of the SHA-256 hash gives 2^64 possible values,
  sufficient to practically eliminate collisions.
- `fillna('NA')` is applied before hashing so records with missing card fields hash
  consistently (missing fields → same uid bucket).
- **Guarantee:** all transactions sharing the same uid are assigned to exactly one bank.
  No uid appears in more than one bank's partition. Verified by `make data-verify`.

---

## Partition Scenarios

### S1 – IID (Independent and Identically Distributed)

- Shuffle all unique uids with fixed seed.
- Assign round-robin to N banks.
- Result: each bank has a similar fraud rate and feature distribution.

### S2 – Feature/Domain Skew

- Cluster uids by their modal (ProductCD, card4, card6) combination.
- Assign clusters to banks so each bank is dominated by different product/card types.
- Result: banks have similar sizes but different feature distributions (real non-IID).

### S3 – Label + Quantity Skew (Dirichlet)

- Draw uid→bank assignment from Dirichlet(α) for each fraud label separately.
- α ∈ {0.1, 0.5, 1.0}: lower α = more extreme skew.
- Result: unequal bank sizes AND very different fraud rates per bank.
- This is the most challenging and realistic scenario.

---

## Temporal Split (per bank)

| Split      | Criterion                             | Resampling allowed? |
|------------|---------------------------------------|---------------------|
| Train      | Earliest 70% of that bank's transactions (by TransactionDT) | Yes (locally) |
| Validation | Next 15% (by TransactionDT)          | No                  |
| Test       | Last 15% (by TransactionDT)          | No                  |

**Rationale:** Temporal splitting reflects real-world model deployment where
the model is trained on historical data and evaluated on future data.
Random splitting would inflate performance by allowing future data to
inform training (temporal leakage).

---

## Per-Bank Statistics (N=5, S1 IID, seed=42)

*Generated and verified by `python data/scripts/verify_splits.py`.*

| Bank | N train | N val | N test | Fraud% | JS distance |
|------|---------|-------|--------|--------|-------------|
| 0    | 86,681  | 18,575| 18,575 | 3.22%  | 0.0066      |
| 1    | 81,272  | 17,415| 17,416 | 4.43%  | 0.0202      |
| 2    | 79,981  | 17,139| 17,139 | 3.26%  | 0.0056      |
| 3    | 87,944  | 18,845| 18,846 | 3.57%  | 0.0016      |
| 4    | 77,498  | 16,607| 16,607 | 3.00%  | 0.0119      |
| Global | 413,376 | 88,581 | 88,583 | 3.50% | 0.0000      |

---

## Per-Bank Statistics (N=5, S2 Feature Skew, seed=42)

| Bank | N train | N val | N test | Fraud% | JS distance |
|------|---------|-------|--------|--------|-------------|
| 0    | 80,788  | 17,312| 17,312 | 3.46%  | 0.0009      |
| 1    | 86,590  | 18,555| 18,555 | 3.22%  | 0.0067      |
| 2    | 78,189  | 16,755| 16,755 | 3.82%  | 0.0072      |
| 3    | 85,885  | 18,404| 18,404 | 3.40%  | 0.0024      |
| 4    | 81,925  | 17,555| 17,556 | 3.64%  | 0.0033      |

---

## Per-Bank Statistics (N=5, S3 Dirichlet α=0.1, seed=42)

*Highest-skew scenario – unequal sizes and very different fraud rates per bank.*

| Bank | N train | N val | N test | Fraud% | JS distance |
|------|---------|-------|--------|--------|-------------|
| 0    | 38,535  | 8,258 | 8,258  | 0.01%  | 0.1318      |
| 1    | 115,700 | 24,793| 24,793 | 0.00%  | 0.1331      |
| 2    | 49      | 11    | 11     | 12.68% | 0.1470      |
| 3    | 259,071 | 55,515| 55,516 | 5.58%  | 0.0426      |
| 4    | 21      | 4     | 5      | 40.00% | 0.4008      |

---

## Per-Bank Statistics (N=5, S3 Dirichlet α=0.5, seed=42)

| Bank | N train | N val | N test | Fraud% | JS distance |
|------|---------|-------|--------|--------|-------------|
| 0    | 88,603  | 18,986| 18,987 | 2.64%  | 0.0212      |
| 1    | 89,669  | 19,215| 19,215 | 0.32%  | 0.1063      |
| 2    | 20,785  | 4,454 | 4,454  | 5.82%  | 0.0469      |
| 3    | 187,832 | 40,250| 40,250 | 4.86%  | 0.0290      |
| 4    | 26,488  | 5,676 | 5,676  | 5.63%  | 0.0435      |

---

## Per-Bank Statistics (N=5, S3 Dirichlet α=1.0, seed=42)

| Bank | N train | N val | N test | Fraud% | JS distance |
|------|---------|-------|--------|--------|-------------|
| 0    | 67,308  | 14,423| 14,424 | 0.71%  | 0.0862      |
| 1    | 175,205 | 37,544| 37,544 | 4.11%  | 0.0135      |
| 2    | 90,842  | 19,466| 19,467 | 2.32%  | 0.0300      |
| 3    | 33,928  | 7,270 | 7,271  | 5.75%  | 0.0457      |
| 4    | 46,093  | 9,877 | 9,878  | 5.93%  | 0.0491      |

---

## Majority-Class Baseline

For any dataset with ~3.5% fraud, a classifier that always predicts
"not fraud" achieves ~96.5% accuracy. This is reported alongside all
accuracy figures to prevent misinterpretation. The primary evaluation
metrics are **PR-AUC** and **ROC-AUC**.

---

## Feature Engineering

Preprocessing is implemented in `fl/preprocessing.py` using a fixed,
public schema stored in `data/vocab/`. The vocabulary is computed once
from the training data (see `data/scripts/compute_vocab.py`) and committed
to the repository, so all banks apply identical encoding without sharing data.

Federated feature scaling (mean/std) is computed via a single secure-sum
round using the same secure aggregation primitive as the model updates.
Each bank contributes (count, sum, sum_sq) per numeric column; the global
aggregate is used to z-score normalise features at all banks.
