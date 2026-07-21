# Model Card: Telco Customer Churn Classifier

## Model Details

- **Model type:** Logistic Regression (scikit-learn `LogisticRegression`, `max_iter=1000`)
- **Version:** trained via `src/train_final_model.py`, artifacts in `models/`
  (`churn_model.joblib`, `scaler.joblib`, `feature_columns.joblib`)
- **Input:** 18 raw customer attributes (demographics, account info, subscribed
  services) — see `app/main.py`'s `CustomerInput` for the exact schema and
  validation bounds
- **Output:** churn probability in `[0, 1]`, plus a boolean prediction at a
  caller-supplied decision threshold (`/predict?threshold=`)
- **Served via:** FastAPI (`app/main.py`), containerized (`Dockerfile`)

### Why Logistic Regression over Random Forest / XGBoost

All three were trained and compared on an identical 80/20 stratified split
(`src/train_models.py`, `src/evaluate_models.py`):

| Model | Train acc. | Test acc. | Overfitting gap | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 80.53% | **79.77%** | **0.76%** | **0.64** | **0.55** | **0.59** |
| Random Forest | 99.79% | 78.28% | 21.51% | 0.62 | 0.48 | 0.54 |
| XGBoost | 93.70% | 77.43% | 16.27% | 0.59 | 0.51 | 0.54 |

Logistic Regression was chosen because it generalizes best (smallest
train/test gap — the tree ensembles memorize training data more than they
learn transferable patterns on this dataset) and wins on every Churn-class
metric, not just accuracy. A secondary but practically important reason:
its coefficients are directly inspectable, which let us debug individual
predictions during manual testing (see "Known Failure Modes" below) — a
capability Random Forest/XGBoost don't offer as directly.

## Intended Use

- **Primary use case:** score existing telecom customers to prioritize
  outreach for a retention program (e.g. discount offers, proactive support
  calls), as explored in the cost-benefit analysis in `src/business_framing.py`.
- **Not intended for:** individual high-stakes decisions about a specific
  customer (e.g. denying service, pricing an individual contract) without
  human review — this is a prioritization tool, not an automated
  decision-maker.
- **Out of scope:** any customer population meaningfully different from the
  training data (different country, different service offerings, different
  pricing era). The model has not been validated on any such population.

## Training Data

- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  (Kaggle), 7,043 customers, single snapshot in time.
- **Class balance:** 26.54% churned / 73.46% retained (moderately
  imbalanced, not extreme).
- Final model retrained on 100% of the 7,043 rows (`src/train_final_model.py`)
  after model selection and evaluation were already finalized on the held-out
  test split — see "Evaluation" below for the metrics that drove that
  selection, which were computed *before* this final refit.

## How Class Imbalance Was Handled

Two things were deliberately **not** done, and one thing was:

- **Not used: resampling (SMOTE, random over/undersampling).** With only a
  ~1:2.8 imbalance ratio (not the extreme 1:100+ cases where resampling
  earns its complexity), and given that Stage 7 already builds an explicit
  cost model around the true class proportions, artificially rebalancing the
  training data would distort the probability estimates the cost model
  depends on being calibrated to the real-world churn rate.
- **Not used: `class_weight='balanced'`.** This would push the model to
  treat both classes as equally important by construction. We instead let
  the *decision threshold* carry that responsibility (see below), which
  keeps the model's probability output meaningful and moves the
  cost-sensitivity into an explicit, auditable, business-driven parameter
  instead of burying it inside training.
- **Used: `stratify=y` in the train/test split** (`src/train_models.py`),
  preserving the 26.54% churn ratio in both splits so evaluation metrics
  aren't distorted by a lucky/unlucky split.
- **Used: evaluating Precision/Recall/F1 on the minority (Churn) class
  specifically** (`src/evaluate_models.py`), instead of relying on overall
  accuracy, which is misleading here — a trivial "predict majority class"
  model already scores 73.46% accuracy without learning anything (see
  `PROJECT_LOG.md` section 7 for the full reasoning).

## Decision Threshold: Why It Isn't a Fixed 0.5 (or a Fixed Anything)

The standard ML default of 0.5 is **not** business-optimal here. In
`src/business_framing.py`, each confusion-matrix cell was assigned a real
dollar cost/benefit (retention offer cost ≈ 20% of `MonthlyCharges`,
retention value ≈ 12 months of `MonthlyCharges`, 30% offer success rate),
and sweeping the threshold showed the profit-maximizing cutoff was much
lower — around **0.05**, because a False Negative (losing a customer we
never attempted to retain) is far more costly than a False Positive (an
unnecessary but cheap retention offer). The breakeven math:

```
breakeven probability = offer_cost / (success_rate × retention_value) ≈ 5.6%
```

Since the base churn rate (26.54%) is well above this breakeven point,
broad targeting is economically rational. `src/sensitivity_analysis.py`
confirmed this conclusion holds (a low threshold stays optimal) across four
different cost/success-rate assumptions, including a pessimistic one.

**Why the API still defaults to 0.5 instead of hardcoding 0.05:** the
"optimal" threshold is a function of business assumptions (offer cost,
success rate) that can change independently of the model — a marketing
team revising the retention-offer budget shouldn't require retraining or
redeploying the model. `app/main.py`'s `/predict` endpoint therefore
exposes `threshold` as a request parameter with 0.5 as a neutral,
ML-conventional default, and the business-derived value (or a re-run of
`sensitivity_analysis.py` under updated assumptions) is meant to be
supplied by the caller.

## Evaluation

See the Model Comparison table above for headline numbers (both at the
default 0.5 threshold). Full confusion matrices are in
`notebooks/figures/confusion_matrices.png`.

**At threshold 0.5, the model misses ~45% of actual churners (Recall
0.55) — this is exactly why the business-optimal threshold (Stage 7)
is far lower than 0.5, not an incidental detail.** A False Negative here
means a customer we never even attempted to retain; the cost-benefit
analysis in `src/business_framing.py` showed that because the retention
offer is cheap relative to the value of keeping a customer, it's worth
tolerating far more false alarms in exchange for catching almost all real
churners. The table below makes that trade-off concrete, computed on the
same test split (`random_state=42`) used throughout Stages 5-6:

| Metric (Churn class) | Threshold 0.5 (default) | Threshold 0.05 (business-recommended) |
|---|---|---|
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[918, 117], [168, 206]]` | `[[370, 665], [6, 368]]` |
| Precision | 0.638 | 0.356 |
| **Recall** | **0.551** | **0.984** |
| F1 | 0.591 | 0.523 |
| Overall accuracy | 0.798 | 0.524 |

Lowering the threshold to 0.05 catches 368 of 374 actual churners
(Recall 98.4%, missing only 6) at the cost of 665 false alarms (Precision
drops to 35.6%) and a much lower overall accuracy (52.4%). This looks
like a worse model by traditional ML metrics — but per the cost model in
Stage 7, it is the more profitable operating point, because a missed
churner is far more expensive than an unnecessary retention offer. This
is the clearest illustration in the whole project of why accuracy (and
even F1) can point to a different answer than what's actually optimal
for the business.

## Model Interpretability (SHAP)

`src/shap_analysis.py` computes exact Shapley values for the deployed
Logistic Regression via `shap.LinearExplainer` (closed-form for linear
models, no sampling approximation needed), against a 100-row background
sample of scaled training data. This generalizes the manual
coefficient-decomposition done during debugging (see the fixed
`MonthlyCharges` sign-flip note below) into a proper, model-agnostic
framework — the same approach works for the tree-based models too, not
just linear ones.

**Global feature importance** (`notebooks/figures/shap_global_importance.png`,
mean |SHAP value| across the test set) confirms the same top drivers found
independently via SQL/EDA in Stages 2-3: `InternetService_Fiber optic`,
`MonthlyCharges`, `tenure`, and `Contract` dominate. The beeswarm plot
(`shap_beeswarm.png`) adds *direction*: it shows visually that high
`MonthlyCharges` (red dots) cluster on the *negative* SHAP side (pushing
away from churn) while low `MonthlyCharges` (blue dots) push toward
churn — a direct visual confirmation of the multicollinearity-driven sign
flip described below, this time with no manual coefficient math required.

**Individual explanations** (`shap_waterfall_lowest_risk.png`,
`shap_waterfall_highest_risk.png`) show the same customer-by-customer
breakdown as before, but now derived from a principled method rather than
`coefficient × scaled_value` by hand. The highest-risk test customer
(87% predicted churn probability) is driven up mainly by low `tenure`
(+1.05) and `InternetService_Fiber optic` (+0.82), while — again — a
high `MonthlyCharges` value pulls the prediction *down* (-0.84) despite
being a "risky-looking" number in isolation.

**Served live via `/explain`:** the same `LinearExplainer`, built once at
API startup against a background sample saved by
`src/train_final_model.py` (`models/shap_background.joblib`), powers a
`/explain` endpoint (see `app/main.py`) that takes the same
`CustomerInput` payload as `/predict` and returns every feature's SHAP
contribution plus the base value, sorted by magnitude. This means the
kind of manual debugging done in Session 11 (decomposing one prediction
by hand to understand a borderline case) is now a first-class,
always-available API capability instead of an ad hoc investigation.

## Known Failure Modes / Limitations

- **[Fixed] Single-request one-hot encoding bug (found via unit tests,
  `tests/test_api.py`).** `encode_customer` used to call
  `pd.get_dummies(..., drop_first=True)` directly on a single-row
  DataFrame. Since a lone request only ever has *one* value per nominal
  column, pandas treated that single value as "the only category" and
  dropped it as the baseline — regardless of whether it matched the
  actual training-time baseline. Concretely, a customer with
  `InternetService="Fiber optic"` and `PaymentMethod="Electronic check"`
  had *both* encoded as all-zero (i.e. silently treated as `DSL` and
  `Bank transfer (automatic)`, the true training baselines) instead of
  their real values. This was serious: the same test customer's predicted
  churn probability changed from **29.0% (wrong) to 64.9% (correct)**
  once fixed. **Importantly, this bug was confined to the live `/predict`
  encoding path** — Stages 5-7 (model training, evaluation, business
  framing) all ran `get_dummies` over the full 7,043-row dataset at once,
  where every category is naturally present, so those results are
  unaffected. Fix: nominal columns are now cast to `pd.Categorical` with
  the exact training-time category list (`NOMINAL_CATEGORIES` in
  `app/main.py`) before encoding, so a single request always produces the
  same columns regardless of which value it has. This is also documented
  as the motivating case for adding `tests/test_api.py` in the first
  place — it was caught by a unit test, not manual testing.
- **[Fixed] SHAP background sample size not bounded by dataset size
  (found via the pipeline integration test, `tests/test_pipeline.py`).**
  `src/train_final_model.py` sampled a fixed `SHAP_BACKGROUND_SIZE=100`
  rows for the `/explain` endpoint's background distribution via
  `rng.choice(..., replace=False)`. This works fine on the real
  7,043-row dataset but raises `ValueError` on any dataset smaller than
  100 rows (e.g. the test suite's 60-row synthetic dataset) since you
  can't sample more rows than exist without replacement. Fixed by
  capping the sample size to `min(SHAP_BACKGROUND_SIZE, len(X_scaled))`.
  Doesn't change behavior on the real dataset at all — purely a
  robustness fix the smaller test dataset happened to expose.
- **Extrapolation beyond training range.** Manual testing surfaced that
  `MonthlyCharges=0` (a value with no real analog in training data, where
  the observed minimum was ~$18.25) produced an unreliable prediction
  right at the 0.5 boundary. `app/main.py` now enforces
  `tenure ∈ [0, 72]` and `MonthlyCharges ∈ (0, 150]` to match the range
  the model actually learned from, rejecting out-of-distribution inputs
  with a 422 rather than silently extrapolating.
- **Coefficients are partial effects, not simple correlations.** The
  `MonthlyCharges` coefficient is *negative* in the fitted model, despite
  `MonthlyCharges` having a *positive* univariate correlation with churn in
  EDA (Stage 3). This is a multicollinearity effect — `MonthlyCharges` is
  correlated with `InternetService`/add-on-service features already in the
  model, which absorb most of its predictive signal. Anyone inspecting
  individual coefficients from this model should not read them as
  standalone "this variable causes more/less churn" statements.
- **Linear decision boundary.** Logistic Regression can't capture
  interaction effects (e.g. "Fiber optic is only risky when combined with
  no tech support") unless they're engineered as explicit features, which
  wasn't done here. This is a plausible reason its Recall plateaus at 55%.
- **Single point-in-time snapshot.** No temporal validation was done (e.g.
  training on an earlier period, testing on a later one) — real-world
  churn drivers (pricing, competition, service quality) can drift, and this
  model has no mechanism to detect or adapt to that.
- **Business-framing assumptions are estimates, not measured values.**
  Offer cost, success rate, and retention value in `src/business_framing.py`
  are reasonable assumptions, not numbers pulled from this specific
  business's actual retention program. The sensitivity analysis shows the
  *qualitative* conclusion (low threshold, positive ROI) is robust, but the
  *exact* dollar figures should not be quoted as guaranteed outcomes.

## Recommendations for Future Work

- Re-run `sensitivity_analysis.py` with real observed retention-offer
  cost/success-rate data once the business has run an actual campaign,
  rather than the estimated assumptions used here.
- Consider feature interactions or a non-linear model if Recall needs to
  improve beyond ~55%, accepting the overfitting risk that requires
  managing (e.g. via regularization tuning or more cross-validation folds
  than the single train/test split used here).
- Add monitoring for input distribution drift in production (e.g. flagging
  when incoming `MonthlyCharges`/`tenure` values cluster near the
  validation bounds), since the model's reliability degrades outside the
  training range it has seen.
