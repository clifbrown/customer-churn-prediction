"""Run from the repository root: python -m src.train.

Select by five-fold training average precision, lock a validation threshold,
then measure holdout performance. All report numbers come from this run.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.preprocessing import get_feature_target, build_pipeline
from src.evaluation import measure, choose_threshold, capacity_metrics, bootstrap_intervals

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates():
    # No resampling or class reweighting: AP and a validation cutoff address
    # the operational imbalance without changing the training class prior.
    return {
        'dummy_prior': build_pipeline(DummyClassifier(strategy='prior')),
        'xgboost': build_pipeline(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                      random_state=SEED, n_jobs=2)),
        'logistic_regression': build_pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        'random_forest': build_pipeline(RandomForestClassifier(n_estimators=200, max_depth=10,
                          min_samples_leaf=5, random_state=SEED, n_jobs=2)),
        'hist_gradient_boosting': build_pipeline(HistGradientBoostingClassifier(max_iter=150,
                          max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
                          early_stopping=False, random_state=SEED)),
    }


def make_splits(X, y):
    train_val, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    train, validation = train_test_split(train_val, test_size=0.25,
                          stratify=y.iloc[train_val], random_state=SEED)
    return {'train': train, 'validation': validation, 'test': test}


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run(data_path=None, output_root=ROOT, bootstrap=500):
    started = time.perf_counter()
    data_path = Path(data_path or ROOT / 'data/raw/telco_churn.csv')
    output_root = Path(output_root)
    model_dir, reports = output_root / 'models', output_root / 'reports'
    figs = reports / 'figures'
    for folder in [model_dir, figs]:
        folder.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    if raw.customerID.isna().any() or raw.customerID.duplicated().any():
        raise ValueError('Training data requires nonmissing, unique customerID values.')
    X, y = get_feature_target(raw)
    splits = make_splits(X, y)
    tr, va, te = (splits[k] for k in ['train', 'validation', 'test'])
    data_audit = {'rows': len(raw), 'columns': len(raw.columns), 'predictors': X.shape[1],
                  'churners': int(y.sum()), 'churn_fraction': float(y.mean()),
                  'duplicate_ids': int(raw.customerID.duplicated().sum()),
                  'blank_total_charges': int(raw.TotalCharges.astype(str).str.strip().eq('').sum()),
                  'splits': {k: {'rows': len(i), 'churners': int(y.iloc[i].sum())} for k, i in splits.items()}}
    print('DATA AUDIT', json.dumps(data_audit), flush=True)
    split_rows = []
    for name, idx in splits.items():
        split_rows.extend({'source_row': int(i), 'customerID': raw.customerID.iloc[i], 'split': name} for i in idx)
    pd.DataFrame(split_rows).to_csv(reports / 'splits.csv', index=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipelines = candidates()
    comparison, cv_folds = {}, []
    for name, pipe in pipelines.items():
        tick = time.perf_counter()
        with threadpool_limits(limits=2):
            scores = cross_validate(pipe, X.iloc[tr], y.iloc[tr], cv=cv,
                         scoring={'ap': 'average_precision', 'auc': 'roc_auc'}, n_jobs=1,
                         error_score='raise')
            pipe.fit(X.iloc[tr], y.iloc[tr])
        comparison[name] = {'cv_average_precision': float(scores['test_ap'].mean()),
                            'cv_ap_std': float(scores['test_ap'].std()),
                            'cv_roc_auc': float(scores['test_auc'].mean()),
                            'cv_auc_std': float(scores['test_auc'].std()),
                            'fit_and_cv_seconds': round(time.perf_counter() - tick, 3)}
        for i, (ap, auc) in enumerate(zip(scores['test_ap'], scores['test_auc'])):
            cv_folds.append({'model': name, 'fold': i+1, 'average_precision': ap, 'roc_auc': auc})
        print(name, json.dumps(comparison[name]), flush=True)
    # This choice never reads test or validation scores.
    selected_name = max(comparison, key=lambda n: comparison[n]['cv_average_precision'])
    selected = pipelines[selected_name]
    with threadpool_limits(limits=2):
        pv = selected.predict_proba(X.iloc[va])[:, 1]
    threshold = choose_threshold(y.iloc[va], pv, target_recall=0.8)
    validation = measure(y.iloc[va], pv, threshold)
    # Lock this selection before computing ANY test metrics. No subsequent refit:
    # the exported estimator is exactly the model whose scores are evaluated.
    print(f'LOCKED selection={selected_name}; threshold={threshold:.8f}', flush=True)
    pd.DataFrame(cv_folds).to_csv(reports / 'cv_folds.csv', index=False)
    ptest = {}
    for name, pipe in pipelines.items():
        with threadpool_limits(limits=2):
            p = pipe.predict_proba(X.iloc[te])[:, 1]
        ptest[name] = p
        comparison[name]['test_at_0_5'] = measure(y.iloc[te], p)
    p = ptest[selected_name]
    selected_metrics = measure(y.iloc[te], p, threshold)
    intervals = bootstrap_intervals(y.iloc[te], p, threshold, repetitions=bootstrap)
    capacity = capacity_metrics(y.iloc[te], p)
    # A pre-specified LR ablation isolates whether the extra ratio helps in CV.
    # It is descriptive and does not alter the already locked candidate selection.
    ablation_pipe = build_pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED), False)
    with threadpool_limits(limits=2):
        ablated = cross_validate(ablation_pipe, X.iloc[tr], y.iloc[tr], cv=cv,
                      scoring='average_precision', n_jobs=1, error_score='raise')['test_score']
    full_fold = np.array([r['average_precision'] for r in cv_folds if r['model']=='logistic_regression'])
    ablation = {'without_spend_cv_ap': float(ablated.mean()),
                'with_spend_cv_ap': comparison['logistic_regression']['cv_average_precision'],
                'paired_fold_differences': (full_fold - ablated).tolist(),
                'mean_difference': float((full_fold-ablated).mean())}
    # Raw-feature permutation keeps all encoded levels together. Importance is
    # predictive association, not causation; correlated features can mask it.
    with threadpool_limits(limits=2):
        # Permute after stateless cleaning: raw consistency validation would
        # otherwise reject intentionally broken service combinations.
        transformed = selected.named_steps['cleaner'].transform(X.iloc[te])
        from sklearn.pipeline import Pipeline
        predictor = Pipeline(selected.steps[1:])
        importance = permutation_importance(predictor, transformed, y.iloc[te],
                     scoring='average_precision', n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({'feature': transformed.columns, 'mean_ap_decrease': importance.importances_mean,
                        'std_ap_decrease': importance.importances_std}).sort_values('mean_ap_decrease', ascending=False)
    imp.to_csv(reports / 'permutation_importance.csv', index=False)
    # Preserve labelled predictions so metrics can be independently recomputed.
    for name, idx, scores in [('test', te, p), ('validation', va, pv)]:
        pd.DataFrame({'source_row': idx, 'customerID': raw.customerID.iloc[idx].to_numpy(),
                      'y_true': y.iloc[idx].to_numpy(), 'churn_score': scores,
                      'flagged': (scores >= threshold).astype(int)}).to_csv(reports / f'{name}_predictions.csv', index=False)
    # Audit subgroup behaviour; small samples are descriptive, not fairness proof.
    group_rows = []
    for col in ['gender', 'SeniorCitizen']:
        for value in sorted(raw.iloc[te][col].unique(), key=str):
            mask = raw.iloc[te][col].eq(value).to_numpy()
            m = measure(y.iloc[te].to_numpy()[mask], p[mask], threshold)
            group_rows.append({'group': col, 'value': value, 'n': int(mask.sum()),
                               'churners': int(y.iloc[te].to_numpy()[mask].sum()),
                               'recall': m['recall'], 'precision': m['precision'],
                               'false_positive_rate': m['fp']/(m['fp']+m['tn'])})
    pd.DataFrame(group_rows).to_csv(reports / 'subgroup_metrics.csv', index=False)
    metrics = {'selected_model': selected_name, 'selection_metric': 'training_5fold_average_precision',
               'seed': SEED, 'data_audit': data_audit, 'comparison': comparison,
               'threshold_policy': 'highest validation cutoff achieving recall >= 0.80',
               'threshold': threshold, 'validation': validation, 'test': selected_metrics,
               'test_95pct_bootstrap_intervals': intervals, 'bootstrap_repetitions': bootstrap,
               'test_top20pct': capacity, 'spend_ablation': ablation,
               'probability_calibrated': False}
    (model_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    joblib.dump(selected, model_dir / 'churn_pipeline.joblib')
    # No duplicate app artifacts and no binaries from the original ZIP are loaded.
    raw.iloc[tr[:10]].drop(columns='Churn').to_csv(output_root / 'data/sample_customers.csv', index=False)
    pd.DataFrame([{'model': name, **{k:v for k,v in entry.items() if not isinstance(v,dict)},
                   **{'test_'+k:v for k,v in entry['test_at_0_5'].items()}}
                   for name, entry in comparison.items()]).to_csv(reports / 'model_comparison.csv', index=False)
    # EDA uses training rows only so the test set is not a design input.
    eda = raw.iloc[tr].copy(); eda['Churn'] = y.iloc[tr]
    for col in ['Contract', 'InternetService']:
        tab = eda.groupby(col).Churn.agg(['count','sum','mean'])
        tab.to_csv(reports / f'training_churn_by_{col}.csv')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    eda.groupby('Contract').Churn.mean().sort_values().plot.barh(ax=axes[0],color='#277e8e')
    axes[0].set(xlabel='Observed churn fraction', title='Training sample: contract type',ylabel='')
    eda.groupby('InternetService').Churn.mean().sort_values().plot.barh(ax=axes[1],color='#277e8e')
    axes[1].set(xlabel='Observed churn fraction',title='Training sample: internet service',ylabel='')
    save_fig(fig, figs/'training_eda.png')
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for name, scores in ptest.items():
        fpr,tpr,_=roc_curve(y.iloc[te],scores); prec,rec,_=precision_recall_curve(y.iloc[te],scores)
        axes[0].plot(fpr,tpr,label=name.replace('_',' '))
        if name == 'dummy_prior':
            axes[1].axhline(y.iloc[te].mean(), label='dummy prior', color='gray', ls='--')
        else:
            axes[1].step(rec,prec,where='post',label=name.replace('_',' '))
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(xlabel='False-positive rate',ylabel='Recall',title='Test ROC curves')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Test precision–recall curves')
    axes[1].legend(fontsize=8); save_fig(fig,figs/'model_curves.png')
    fig, ax = plt.subplots(figsize=(5,4))
    cm=np.array([[selected_metrics['tn'],selected_metrics['fp']],[selected_metrics['fn'],selected_metrics['tp']]])
    ax.imshow(cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black',fontsize=16)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Not flagged','Flagged'],yticklabels=['No churn','Churn'],
           xlabel='Model decision',ylabel='Observed label',title='Test confusion matrix · validation cutoff')
    save_fig(fig,figs/'confusion_matrix.png')
    fig, ax=plt.subplots(figsize=(6,4))
    for name in ['dummy_prior',selected_name]:
        frac,mean=calibration_curve(y.iloc[te],ptest[name],n_bins=8,strategy='quantile')
        ax.plot(mean,frac,'o-',label=name.replace('_',' '))
    ax.plot([0,1],[0,1],'--',color='gray');ax.legend(fontsize=8)
    ax.set(xlabel='Mean predicted score',ylabel='Observed churn fraction',title='Calibration diagnostic · not recalibrated')
    save_fig(fig,figs/'calibration.png')
    fig,ax=plt.subplots(figsize=(8,5)); top=imp.head(10).iloc[::-1]
    ax.barh(top.feature,top.mean_ap_decrease,xerr=top.std_ap_decrease,color='#277e8e')
    ax.axvline(0,color='gray',lw=0.7);ax.set(xlabel='Average precision decrease after permutation',title='Predictive importance · five permutations')
    save_fig(fig,figs/'permutation_importance.png')
    params={name:{k:v for k,v in pipe.named_steps['model'].get_params().items()} for name,pipe in pipelines.items()}
    manifest={'run_utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
              'platform':platform.platform(),'data_sha256':sha256(data_path),
              'seed':SEED,'model_parameters':params,'selection_metric':metrics['selection_metric'],
              'runtime_seconds':round(time.perf_counter()-started,3),
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','joblib','threadpoolctl','xgboost']},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'src').glob('*.py'))}}
    (reports/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    from src.reporting import build_report
    build_report(output_root)
    print('TEST',json.dumps(selected_metrics),flush=True)
    print('CAPACITY',json.dumps(capacity),flush=True)
    print('FINISHED',manifest['runtime_seconds'],'seconds',flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/telco_churn.csv')
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args()
    run(args.data,args.output_root)
