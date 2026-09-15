"""Dependency-light local dashboard: python -m app.app (localhost only)."""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from src.predict import load_artifacts, score_customers
from src.preprocessing import CATEGORIES, NUMERIC_RAW, RAW_FEATURES

ROOT=Path(__file__).resolve().parents[1]
MAX_BODY=2_000_000
MAX_ROWS=5000


def make_handler(root=ROOT):
    root=Path(root)
    model,metrics=load_artifacts(root)
    class Handler(BaseHTTPRequestHandler):
        def send(self,body,status=200,mime='application/json'):
            if isinstance(body,dict):body=json.dumps(body,allow_nan=False).encode()
            elif isinstance(body,str):body=body.encode()
            self.send_response(status)
            self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':
                return self.send((root/'app/static/index.html').read_bytes(),mime='text/html')
            if path=='/api/summary':
                return self.send({'metrics':metrics,'categories':CATEGORIES,'numeric':NUMERIC_RAW,
                    'features':RAW_FEATURES,
                    'example':pd.read_csv(root/'data/sample_customers.csv').iloc[0].to_dict(),
                    'importance':pd.read_csv(root/'reports/permutation_importance.csv').head(10).to_dict('records')})
            if path=='/sample.csv':
                return self.send((root/'data/sample_customers.csv').read_bytes(),mime='text/csv')
            figures={f'/figures/{p.name}':p for p in (root/'reports/figures').glob('*.png')}
            if path in figures:return self.send(figures[path].read_bytes(),mime='image/png')
            self.send({'error':'Not found.'},404)

        def do_POST(self):
            if urlparse(self.path).path!='/api/score':return self.send({'error':'Not found.'},404)
            # Reject cross-origin browser submissions to this local service.
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://{self.headers.get("Host")}':
                return self.send({'error':'Cross-origin requests are not allowed.'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY:raise ValueError('Request must be between 1 byte and 2 MB.')
                payload=json.loads(self.rfile.read(size))
                if 'csv' in payload:
                    frame=pd.read_csv(io.StringIO(payload['csv']))
                else:
                    frame=pd.DataFrame(payload['rows'])
                if len(frame)>MAX_ROWS:raise ValueError('Please upload at most 5,000 rows.')
                out=score_customers(frame,model,metrics['threshold'])
                # Missing input fields remain blank in the CSV; JSON uses null.
                records=json.loads(out.to_json(orient='records'))
                return self.send({'rows':records,'csv':out.to_csv(index=False),
                    'count':len(out),'flagged':int(out.flagged_for_review.sum()),
                    'threshold':metrics['threshold']})
            except (ValueError,KeyError,TypeError,pd.errors.ParserError) as exc:
                return self.send({'error':str(exc)},400)

        def log_message(self,format,*args):
            # Do not log customer payloads.
            super().log_message(format,*args)
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8501)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler())
    print(f'Open http://127.0.0.1:{args.port} — Ctrl+C to stop.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
