"""Meaningful regression checks for data integrity, evaluation, and local serving."""
import io
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.preprocessing import clean, get_feature_target, build_pipeline, validate_features
from src.evaluation import choose_threshold, measure, capacity_metrics
from src.predict import load_artifacts, score_customers
from src.train import make_splits
from app.app import make_handler

ROOT=Path(__file__).resolve().parents[1]


class DataAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(ROOT/'data/raw/telco_churn.csv')
        cls.X,cls.y=get_feature_target(cls.raw)
        cls.pipe,cls.metrics=load_artifacts()

    def test_splits_are_disjoint_complete_and_repeatable(self):
        a=make_splits(self.X,self.y);b=make_splits(self.X,self.y)
        self.assertEqual(set(np.concatenate(list(a.values()))),set(range(len(self.y))))
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertFalse(set(a['train']) & set(a['validation']))
        self.assertFalse(set(a['train']) & set(a['test']))
        self.assertFalse(set(a['validation']) & set(a['test']))

    def test_blank_bills_at_zero_tenure_have_finite_spend(self):
        x=self.X.head(1).copy();x['tenure']=0;x['TotalCharges']=' '
        out=clean(x)
        self.assertEqual(out.TotalCharges.iloc[0],0)
        self.assertEqual(out.AvgMonthlySpend.iloc[0],out.MonthlyCharges.iloc[0])

    def test_missing_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):
            validate_features(self.X.drop(columns='Contract'))
        for value in ['not a number',-5,np.inf]:
            x=self.X.head(1).copy();x['MonthlyCharges']=value
            with self.assertRaises(ValueError):validate_features(x)
        x=self.X.head(1).copy();x['Contract']='imaginary plan'
        with self.assertRaisesRegex(ValueError,'Unsupported'):validate_features(x)

    def test_inconsistent_service_profile_is_rejected(self):
        x=self.X.head(1).copy();x['InternetService']='No';x['OnlineSecurity']='Yes'
        with self.assertRaisesRegex(ValueError,'agree'):validate_features(x)

    def test_numeric_missing_is_imputed_and_training_statistics_are_unchanged(self):
        x=self.X.iloc[:100].copy();y=self.y.iloc[:100]
        pipe=build_pipeline(LogisticRegression(max_iter=2000)).fit(x,y)
        imputer=pipe.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer']
        before=imputer.statistics_.copy()
        query=x.head(2).copy();query['MonthlyCharges']=np.nan;query['TotalCharges']=np.nan
        self.assertTrue(np.isfinite(pipe.predict_proba(query)).all())
        np.testing.assert_array_equal(before,imputer.statistics_)
        np.testing.assert_allclose(before[0],np.median(x.tenure))

    def test_target_accepts_labels_or_binary_and_rejects_unknown(self):
        raw=self.raw.head(3).copy();raw['Churn']=[1,0,1]
        self.assertEqual(get_feature_target(raw)[1].tolist(),[1,0,1])
        raw['Churn']=['Yes','Unknown','No']
        with self.assertRaises(ValueError):get_feature_target(raw)

    def test_pipeline_serialisation_and_batch_id_order(self):
        rows=self.raw.head(5).drop(columns='Churn').copy()
        rows.customerID=['z','a','a','q','b']  # Batch duplicate IDs are retained, not silently removed.
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib';joblib.dump(self.pipe,path);restored=joblib.load(path)
            a=score_customers(rows,self.pipe,self.metrics['threshold'])
            b=score_customers(rows,restored,self.metrics['threshold'])
            np.testing.assert_allclose(a.churn_score,b.churn_score)
            self.assertEqual(a.customerID.tolist(),rows.customerID.tolist())

    def test_threshold_minimises_nested_contact_set_at_target_recall(self):
        y=np.array([1,1,1,1,0,0]);p=np.array([.9,.8,.8,.1,.85,.2])
        threshold=choose_threshold(y,p,.75)
        self.assertEqual(threshold,.8)
        self.assertGreaterEqual(measure(y,p,threshold)['recall'],.75)
        self.assertLess(measure(y,p,np.nextafter(threshold,1))['recall'],.75)

    def test_persisted_predictions_reproduce_reported_metrics(self):
        pred=pd.read_csv(ROOT/'reports/test_predictions.csv')
        expected=self.metrics['test'];actual=measure(pred.y_true,pred.churn_score,self.metrics['threshold'])
        for key,value in actual.items():self.assertAlmostEqual(value,expected[key],places=10)
        raw=self.raw.iloc[pred.source_row].drop(columns='Churn')
        fresh=score_customers(raw,self.pipe,self.metrics['threshold'])
        np.testing.assert_allclose(fresh.churn_score,pred.churn_score,rtol=1e-10)
        cap=capacity_metrics(pred.y_true,pred.churn_score)
        self.assertEqual(cap,self.metrics['test_top20pct'])

    def test_model_selection_uses_cv_not_test_and_threshold_uses_validation(self):
        m=self.metrics
        winner=max(m['comparison'],key=lambda n:m['comparison'][n]['cv_average_precision'])
        self.assertEqual(winner,m['selected_model'])
        v=pd.read_csv(ROOT/'reports/validation_predictions.csv')
        self.assertAlmostEqual(choose_threshold(v.y_true,v.churn_score),m['threshold'],places=12)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler())
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        cls.example=pd.read_csv(ROOT/'data/sample_customers.csv')

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def post(self,payload):
        req=urllib.request.Request(self.url+'/api/score',data=json.dumps(payload).encode(),
                                   headers={'Content-Type':'application/json'})
        return urllib.request.urlopen(req)

    def test_dashboard_assets_and_metrics(self):
        for path in ['/','/api/summary','/sample.csv','/figures/confusion_matrix.png']:
            with urllib.request.urlopen(self.url+path) as r:self.assertEqual(r.status,200)

    def test_single_and_csv_scoring_agree(self):
        with self.post({'rows':self.example.head(1).to_dict('records')}) as r:a=json.load(r)
        with self.post({'csv':self.example.head(1).to_csv(index=False)}) as r:b=json.load(r)
        self.assertAlmostEqual(a['rows'][0]['churn_score'],b['rows'][0]['churn_score'])
        self.assertEqual(pd.read_csv(io.StringIO(b['csv'])).customerID.iloc[0],self.example.customerID.iloc[0])

    def test_bad_schema_returns_actionable_error(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:self.post({'rows':[{'tenure':2}]})
        self.assertEqual(caught.exception.code,400)
        self.assertIn('Missing required',json.load(caught.exception)['error'])

    def test_unknown_paths_do_not_serve_files(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(self.url+'/models/churn_pipeline.joblib')
        self.assertEqual(caught.exception.code,404)


if __name__=='__main__':unittest.main()
