# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single worked lab solution for **DSA 8401 Applied Machine Learning (Chapter 2)**: a
synthetic mobile-money transaction dataset used to teach messy-data auditing and
leakage-safe feature engineering. There is no application code, package, or test suite —
everything lives in one Jupyter notebook plus its input data.

| Path | What it is |
|---|---|
| `Data/mobile_money_statements.csv` | Synthetic mobile-money transactions (~183k rows, 18 columns, `is_fraud` target) |
| `Data/mobile_transt.txt` | Data dictionary for the CSV |
| `notebook/lab2_solution.ipynb` | The full worked solution (31 cells) |

## Running it

```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook notebook/lab2_solution.ipynb
```

The data-load cell (`raw = pd.read_csv(r"...")`) currently hardcodes an absolute Windows
path from the original author's machine, not a relative path to `Data/mobile_money_statements.csv`
in this repo. Fix that path before running the notebook end to end.

## Notebook structure (4 tasks, in order)

1. **Audit** — missingness per column, exact duplicates, validity-rule violations, and an
   MCAR / MAR / MNAR classification per incomplete column (`agent_id` = MAR, `device_id` =
   MCAR, `gps_lat`/`gps_lon` = MNAR).
2. **Clean & aggregate**:
   - Deduplicate exactly.
   - Parse the string `amount` field (formats like `"1,500/-"`, `"(227/-)"`, `"4,742/- Dr"`)
     into a signed numeric via `parse_amount()` — negative sign, `(...)`, or `Dr` suffix all
     mean debit.
   - Parse the three `txn_time` formats into a real timestamp column `ts`.
   - Resolve entities to a canonical `customer_id`: **`msisdn` is not identity** (multi-SIM
     customers own several numbers); `reg_id` (KYC registration id), cleaned via
     `.strip().upper()`, is the correct linking key. Grouping on `msisdn` instead would
     over-count customers and leak a customer's rows across CV folds.
   - Build customer-level RFM + ratio + cyclical features **strictly as-of `SCORING_TS`**
     (`2026-08-01`) — all aggregates use only rows with `ts < SCORING_TS`, enforced before
     feature construction so nothing is computed from future data.
3. **Hunt the leak** — a correlation screen plus a lineage argument identifies
   `manual_review_score` and `settlement_status` as post-outcome fields: they are only
   written *after* a transaction is scored, so including them in training is leakage, not
   signal.
4. **Pipeline** — a single `ColumnTransformer` + `Pipeline` (`build_model(...)`) evaluated
   with `GroupKFold` grouped by `customer_id` (never plain `KFold` — see the entity
   resolution point above), scored once with the leaky columns (`LEAK_NUM`/`LEAK_CAT`)
   excluded and once with them included, to show the AUC inflation leakage causes.

**Headline result:** ROC-AUC ≈ 0.63 on the honest feature set vs. ≈ 1.00 with the leaky
columns added — the gap is illustrating pure leakage, not a modeling improvement to chase.

## Working in this repo

- When editing the notebook, preserve the four-task structure and the ordering constraint
  that `ts < SCORING_TS` filtering happens *before* any customer-level aggregation — that
  ordering is the entire point of the leakage-safety exercise.
- `parse_amount()` and the `txn_time` multi-format parsing are the two "messy data" set
  pieces; if you touch them, re-check `amount parse failures` / equivalent counts printed
  right after, since a regression there silently drops rows to NaN instead of erroring.
- Any new customer-level feature must be computed only from rows with `ts < SCORING_TS` and
  must not be a post-outcome field (i.e., not something written after a transaction is
  scored/settled), or it reintroduces the exact leakage this lab is built to demonstrate.
- Cross-validation must stay grouped by `customer_id` (`GroupKFold`/`groups=groups`); do not
  swap in a non-grouped CV strategy.
