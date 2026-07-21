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

See the Model Comparison table above for headline numbers. Full confusion
matrices are in `notebooks/figures/confusion_matrices.png`. Key caveat:
even the best model misses ~45% of actual churners at the 0.5 threshold
(Recall 0.55) — this is a real limitation, not just a tuning artifact, and
is the direct motivation for the threshold-lowering strategy above rather
than treating 55% Recall as acceptable.

## Known Failure Modes / Limitations

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
