"""Write a readable report directly from computed metrics, never guessed scores."""
from pathlib import Path
import json
import pandas as pd


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def build_report(root):
    root=Path(root)
    m=json.loads((root/'models/metrics.json').read_text())
    manifest=json.loads((root/'reports/run_manifest.json').read_text())
    test=m['test']; val=m['validation']; cap=m['test_top20pct']; audit=m['data_audit']
    ci=m['test_95pct_bootstrap_intervals']; default=m['comparison'][m['selected_model']]['test_at_0_5']
    rows=[]
    for name,v in sorted(m['comparison'].items(),key=lambda pair:pair[1]['cv_average_precision'],reverse=True):
        t=v['test_at_0_5']
        rows.append([name,f"{v['cv_average_precision']:.4f}",f"{v['cv_ap_std']:.4f}",
                     f"{t['roc_auc']:.4f}",f"{t['average_precision']:.4f}",f"{t['f1']:.4f}"])
    comparison=markdown_table(['Model','Training CV AP','Fold SD','Test ROC-AUC','Test AP','Test F1 at 0.5'],rows)
    policy=markdown_table(['Policy','Flagged','True positives','False positives','Missed churners','Precision','Recall'],[
        ['Default 0.5',default['flagged'],default['tp'],default['fp'],default['fn'],f"{default['precision']:.1%}",f"{default['recall']:.1%}"],
        [f"Validation cutoff {m['threshold']:.4f}",test['flagged'],test['tp'],test['fp'],test['fn'],f"{test['precision']:.1%}",f"{test['recall']:.1%}"],
        ['Top 20% capacity',cap['contacts'],cap['churners_captured'],cap['contacts']-cap['churners_captured'],audit['splits']['test']['churners']-cap['churners_captured'],f"{cap['precision']:.1%}",f"{cap['recall']:.1%}"]])
    imp=pd.read_csv(root/'reports/permutation_importance.csv').head(6)
    importance=markdown_table(['Feature','Mean AP decrease','Shuffle SD'],[
        [r.feature,f'{r.mean_ap_decrease:.4f}',f'{r.std_ap_decrease:.4f}'] for r in imp.itertuples()])
    group=pd.read_csv(root/'reports/subgroup_metrics.csv')
    groups=markdown_table(['Group','Value','Test rows','Observed churners','Recall','Precision','False-positive rate'],[
        [r.group,r.value,r.n,r.churners,f'{r.recall:.1%}',f'{r.precision:.1%}',f'{r.false_positive_rate:.1%}'] for r in group.itertuples()])
    ab=m['spend_ablation']
    result=f'''# Churn case study: executed results and interpretation

Author: Clifton Brown Ommila · Run: {manifest['run_utc']} · Python {manifest['python']}

## What was actually executed?

The pipeline was fitted from the supplied CSV. Five candidates were evaluated with five-fold stratified training cross-validation, producing 25 validation scores. A separate five-fold logistic-regression ablation tested the engineered spending ratio. Every table here is generated from saved metrics and predictions. No original serialized model or original result image was reused.

The dataset contains **{audit['rows']:,} customer records**, {audit['columns']} columns (19 raw predictors, one ID, and the target), and **{audit['churners']:,} churn labels ({audit['churn_fraction']:.2%})**. This is the sample's prevalence, not an annual telecom industry churn rate. There are no duplicate customer IDs and {audit['blank_total_charges']} blank cumulative bills. The blanks at zero tenure become zero; other numeric missing values use medians learned inside each training fold.

## Evaluation design

- Training: {audit['splits']['train']['rows']:,} rows. All candidates use identical five-fold stratified splits; each fold learns its own imputers, scaler, and encoder.
- Validation: {audit['splits']['validation']['rows']:,} rows. Used only to choose the review cutoff for the selected model.
- Test: {audit['splits']['test']['rows']:,} rows, including {audit['splits']['test']['churners']} churners. Used after model and cutoff choices are locked.
- Fixed split seed: {m['seed']}. IDs and target are excluded from features. There is no resampling or class reweighting.
- Candidate hyperparameters are fixed, not a large search. The selection criterion is average precision (AP), chosen because positive-class ranking matters under imbalance.

The original project already inspected this public dataset and an earlier test split. The new split enforces separation inside this implementation but does **not** make this an unseen external benchmark. There is no timestamped evaluation, prospective campaign, or independent customer population.

## Model comparison

{comparison}

**Selected: `{m['selected_model']}`**, using the largest mean training CV AP. The test scores do not change this choice. The models are close in CV and the fold standard deviations overlap; that is not a formal significance test or evidence of a decisive winner. A simple model can be a defensible portfolio result.

The prior-only baseline gives every profile the training prevalence. Its ROC-AUC is 0.5 and its test AP is approximately test prevalence. High majority-class accuracy alone would not identify churners.

## What do the selected model's scores mean?

- **ROC-AUC {test['roc_auc']:.4f}**, 95% bootstrap interval **{ci['roc_auc'][0]:.4f}–{ci['roc_auc'][1]:.4f}**: the model generally ranks an observed churner above a non-churner. This is not the percentage of customers classified correctly.
- **Average precision {test['average_precision']:.4f}**, interval **{ci['average_precision'][0]:.4f}–{ci['average_precision'][1]:.4f}**: precision summarised over recall levels. This is not precision at the operational cutoff, and it is not trapezoidal PR-AUC.
- **Brier score {test['brier_score']:.4f}**, versus **{m['comparison']['dummy_prior']['test_at_0_5']['brier_score']:.4f}** for the prior baseline: average squared error of probability estimates, where lower is better. Better Brier score alone does not prove good calibration; inspect the calibration plot.
- Scores are **not recalibrated**. No claim is made that a score of 0.8 guarantees an 80% churn frequency in a different population.

Intervals use {m['bootstrap_repetitions']} bootstrap resamples of test rows at a fixed fitted model and cutoff. They omit retraining uncertainty, threshold-selection uncertainty, and future distribution shift.

![Model comparison](figures/model_curves.png)

## Decision threshold and workload

Illustrative objective: find the **highest validation cutoff reaching at least 80% recall**. Because higher cutoffs produce nested smaller flagged sets, this minimises contact volume subject to that validation target. It is not a revenue-optimised threshold.

The chosen cutoff is **{m['threshold']:.6f}**. Validation recall is {val['recall']:.1%}; test recall is {test['recall']:.1%}. Meeting the validation target does not guarantee meeting it elsewhere.

{policy}

At the validation cutoff, **{test['tp']} of {audit['splits']['test']['churners']} observed churners are flagged** and **{test['fn']} are missed**. However, **{test['fp']} flagged profiles did not churn**. The team would review **{test['flagged']} of {audit['splits']['test']['rows']:,} profiles ({test['flagged_fraction']:.1%})**, with {test['precision']:.1%} precision. That is the practical workload behind the recall score.

If capacity is limited to the top 20% instead, reviewing {cap['contacts']} profiles captures {cap['churners_captured']} churners ({cap['recall']:.1%} recall), with {cap['precision']:.1%} precision and **{cap['lift_over_random']:.2f}×** the random-selection positive rate. This is a ranking diagnostic, not an estimate of prevented churn or savings. The top-20% policy was pre-specified for comparison; it does not change the deployed cutoff.

![Test confusion matrix](figures/confusion_matrix.png)
![Calibration diagnostic](figures/calibration.png)

## Feature engineering: was the spending ratio useful?

With `AvgMonthlySpend`, logistic regression CV AP is {ab['with_spend_cv_ap']:.4f}; without it, {ab['without_spend_cv_ap']:.4f}. The mean paired fold difference is **{ab['mean_difference']:+.4f}**. This small, dataset-specific comparison does not establish a reliable improvement. The feature is retained as a documented experiment; no large performance benefit is claimed. Its incremental value may be limited because it overlaps with existing billing fields.

## Explaining model reliance

{importance}

These are **permutation importance** results for the selected model on the test rows, using AP and five shuffles. They measure predictive reliance, not causal effects or per-customer reasons. Error bars are shuffle standard deviations. Importance was computed after stateless cleaning so the engineered ratio is an independently permuted column; its dependence on cumulative charges and tenure can make these combinations unrealistic. Correlated predictors can share or mask importance.

The original SHAP section was replaced with this directly reproducible analysis. No SHAP-derived claim is made. Training-only contract and internet-service summaries are saved separately so descriptive segment associations can be checked.

![Feature importance](figures/permutation_importance.png)
![Training associations](figures/training_eda.png)

## Subgroup error check

{groups}

These are descriptive test-set error rates, without subgroup uncertainty intervals or a causal adjustment. Smaller groups can have unstable estimates. This is an initial audit, not fairness certification. Demographic predictors remain in the benchmark model; operational inclusion would require a purpose-specific review and a validated comparison with a model excluding them.

## Realistic next steps

1. Define the prediction date and future churn horizon; verify that every billing and contract field exists at scoring time. This snapshot cannot establish that temporal condition.
2. Validate on later customers from the actual service, including calibration and subgroup errors.
3. Agree on review capacity, contact cost, and expected response rates before choosing a commercial threshold.
4. Test outreach in a randomised pilot against a control group. Associations with contracts or add-ons do not prove those changes prevent churn.
5. Monitor data quality, score distributions, outcomes, and service performance before any production deployment.

## Reproduction evidence

- Dataset SHA-256: `{manifest['data_sha256']}`.
- [`run_manifest.json`](run_manifest.json): exact package versions, hyperparameters, seed, runtime, and training-source hashes.
- [`test_predictions.csv`](test_predictions.csv) and [`validation_predictions.csv`](validation_predictions.csv): source row, customer ID, true label, score, and flag.
- [`splits.csv`](splits.csv) and [`cv_folds.csv`](cv_folds.csv): split membership and individual CV scores.
- [`training.log`](training.log), [`tests.log`](tests.log), and the executed notebook provide execution evidence.
- The exported pipeline is fitted on training rows only, not refitted on validation or test rows after evaluation.

## Sources and scope

The original archive identifies the [IBM Telco example repository](https://github.com/IBM/telco-customer-churn-on-icp4d) as its source. IBM describes its Telco sample as a [fictional company dataset](https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn). Genuine execution here means genuine calculations on the supplied sample, not verified commercial customer outcomes. The supplied file is retained unchanged and fingerprinted; its upstream download history is not independently attested.

Metric definitions follow [scikit-learn's evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html); calibration cautions follow its [calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
'''
    (root/'reports/RESULTS.md').write_text(result)
    return result


if __name__=='__main__':
    build_report(Path(__file__).resolve().parents[1])
