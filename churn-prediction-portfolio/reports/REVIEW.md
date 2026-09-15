# Review of the original project and implemented improvements

This review concerns the uploaded churn-prediction-portfolio archive. Its original CSV was preserved; original model binaries were not executed or used to supply new results.

| Finding in original project | Why it matters | Implemented change |
| --- | --- | --- |
| Best candidate selected with test ROC-AUC | The test set participated in model selection | Candidate choice uses five-fold training AP; validation selects a cutoff; test follows both |
| One preprocessor instance reused across fitted candidates | Shared mutable estimator state can couple separately saved pipelines | Every candidate constructs an independent raw-data pipeline |
| Cleaning and ratio construction occur outside the persisted estimator | A caller must reproduce hidden preprocessing correctly | Cleaning is a serialisable sklearn pipeline step |
| Models, helper code, and figures copied into `app/` | Retraining the root does not refresh what the app loads | One root artifact directory and shared `src` modules |
| Hard-coded 0.5 prediction threshold plus inconsistent batch risk tiers | Different interfaces communicate different decisions | One saved validation cutoff used by CLI and dashboard |
| App invents cumulative bills as current charge × tenure | Changed prices and billing history are lost | Single-profile input requests observed TotalCharges |
| Limited schema checks and contradictory service choices allowed | Bad data can silently distort scores or crash the app | Numeric, categorical, missing-column, and service-consistency validation |
| Batch input drops customer IDs | Output is harder to reconcile with the source | Original identifiers and row order are retained |
| Claims of annual 26% industry churn and acquisition cost multiples | These are not established by the uploaded data | Replaced with the measured sample prevalence and explicit scope |
| Model explanations described as proof of causal drivers | Prediction does not establish effective interventions | Global permutation reliance and testable business hypotheses |
| SHAP axes always labelled as probability impact | Linear/boosted explanations can instead use log-odds or margins | Replaced with clearly labelled AP-based permutation importance |
| Notebook reads pre-existing metrics rather than fitting all models | A reader cannot connect displayed results to execution | Notebook runs the shared trainer and verifies saved scores |
| No genuine baseline or feature ablation | Additional complexity was not justified empirically | Prior-only baseline plus paired LR spending-ratio ablation |
| No uncertainty or explicit contact workload | Single metrics hide practical trade-offs | Bootstrap intervals, confusion counts, and fixed top-20% capacity diagnostic |
| Broad dependency ranges and no tests | Environment changes can undermine reproducibility | Pinned tested dependencies, manifest, and 14 regression checks |
| Live deployment claims without a supplied live URL | Overstates demonstrated status | Clearly described, tested local dashboard |
| Unverified benchmark comparisons and personal profile placeholders | Weakens credibility | Removed benchmarks, generic profile links, and unsupported employment claims |

## Design choices retained or changed

- Retained the dataset, logistic regression, random forest, XGBoost, spending-ratio experiment, notebook walkthrough, and single/batch scoring use cases.
- Added histogram gradient boosting as an additional fixed candidate and a prior-only baseline.
- Used AP rather than ROC-AUC for selection to focus on the positive-class ranking task. Both remain reported.
- Removed class weighting in this experiment; models learn on the original label distribution and a validation cutoff defines the review trade-off. This does not itself guarantee calibrated probabilities.
- Replaced the original Streamlit app with a small local dashboard using Python's standard HTTP server. This is tested locally and has fewer runtime dependencies. It is not hardened for public hosting.
- Replaced SHAP with permutation importance instead of preserving unexplained plots or claiming probability-scale effects incorrectly.
- Package downloads initially stalled and later completed, allowing XGBoost to remain in the final comparison. Standard separate-kernel Jupyter execution failed because the environment rejected its socket setup; the delivered notebook was fully run in-process with IPython instead.

## What has not been solved

A random split of a previously explored public sample is not a new external test. Feature timestamps and a prospective horizon are unavailable. Calibration on new customers, economic impact, fairness, and causal responses to outreach remain unestablished. There is no real-time feature pipeline, authentication, monitoring, or public production deployment.

The original saved scores were not independently validated and should not be compared as a controlled before/after experiment: the revised split, metric, class-weighting policy, and some model settings differ. The aim is a trustworthy portfolio study, not a claim that this revision increases the headline score.

## Verification completed

- All 11 notebook code cells executed in order with actual outputs and no errors.
- All 14 automated data, model, and HTTP checks passed.
- Exported holdout scores were regenerated from the model and raw source rows; report metrics matched recomputation.
- Source checksums matched the final training manifest.
- Generated model-comparison and importance figures were visually inspected.
- The CSV prediction CLI was executed successfully on the ten-row example.
- Dashboard JavaScript passed syntax checking. HTTP routes, single and batch scoring, error responses, and asset access were exercised.
- Browser rendering and clicking were not verified: the browser executable was unavailable and its download timed out. The local dashboard has no live deployment.
