"""Score raw CSV profiles with the same saved pipeline and decision cutoff."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.preprocessing import validate_features

ROOT = Path(__file__).resolve().parents[1]


def load_artifacts(root=ROOT):
    root=Path(root)
    # Load only this project's freshly generated artifact, never user uploads.
    path=root/'models/churn_pipeline.joblib'
    if not path.exists():
        raise FileNotFoundError('Model missing. Run python -m src.train first.')
    return joblib.load(path), json.loads((root/'models/metrics.json').read_text())


def score_customers(frame, pipeline, threshold):
    validate_features(frame)
    with threadpool_limits(limits=2):
        scores=pipeline.predict_proba(frame)[:,1]
    if not np.isfinite(scores).all():
        raise ValueError('Model returned a non-finite score.')
    # Keep the original IDs and input row order to enable reliable CRM joins.
    output=frame.copy()
    output['churn_score']=scores
    output['flagged_for_review']=(scores>=threshold).astype(int)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    model,metrics=load_artifacts()
    result=score_customers(pd.read_csv(args.input),model,metrics['threshold'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Scored {len(result)} rows; output: {args.output}')
