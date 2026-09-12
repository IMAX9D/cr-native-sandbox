"""Read-only, loopback league monitor. No model loads, cloud writes or training controls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlsplit, parse_qs

MAX_BYTES = 2_000_000


def validate(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('requires league-monitor schema_version=1')
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): raise ValueError('non-finite value')
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        if isinstance(value, list):
            for v in value: finite(v)
    finite(data)
    models=data.get('models',[]); matches=data.get('matchups',[])
    for field in ('models','matchups','qualifications','events'):
        if not isinstance(data.get(field,[]),list):raise ValueError('list required: '+field)
    if len(models)>64 or len(matches)>20000: raise ValueError('snapshot exceeds UI limits')
    ids=[]
    for model in models:
        if not isinstance(model,dict):raise ValueError('model object required')
        key=model.get('id')
        if not isinstance(key,str) or not key or len(key)>120:raise ValueError('invalid model id')
        ids.append(key)
        if not isinstance(model.get('history',[]),list) or len(model.get('history',[]))>2000:raise ValueError('invalid history')
        if model.get('role') not in ('base','historical','champion','candidate','exploiter',None):raise ValueError('invalid role')
        for point in model.get('history',[]):
            if not isinstance(point,dict):raise ValueError('invalid rating point')
            if not all(isinstance(point.get(k),(int,float)) and not isinstance(point[k],bool) for k in ('step','elo')):
                raise ValueError('invalid rating point')
        if model.get('elo') is not None and (isinstance(model['elo'],bool) or not isinstance(model['elo'],(int,float))):
            raise ValueError('invalid Elo')
    if len(set(ids))!=len(ids):raise ValueError('duplicate model id')
    for model in models:
        parent=model.get('parent')
        if parent and parent not in ids:raise ValueError('unknown parent')
        chain={model['id']}
        while parent:
            if parent in chain:raise ValueError('cyclic lineage')
            chain.add(parent);parent=next(m.get('parent') for m in models if m['id']==parent)
    keys=set()
    for row in matches:
        if not isinstance(row,dict):raise ValueError('matchup object required')
        if row.get('a') not in ids or row.get('b') not in ids or row['a']==row['b']:
            raise ValueError('invalid matchup models')
        if row.get('scope') not in ('evaluation','training'):raise ValueError('matchup scope required')
        if not isinstance(row.get('deck'),str) or not row['deck']:raise ValueError('deck required')
        key=(row['scope'],*sorted((row['a'],row['b'])),row['deck'])
        if key in keys:raise ValueError('duplicate aggregate matchup')
        keys.add(key)
        for k in ('wins','draws','losses'):
            if type(row.get(k)) is not int or row[k]<0:raise ValueError('invalid game counts')
    if (matches or any(m.get('elo') is not None or m.get('history') for m in models)) and not data.get('evaluation_id'):
        raise ValueError('comparison cohort required')
    qkeys=set()
    for q in data.get('qualifications',[]):
        if not isinstance(q,dict) or q.get('model') not in ids or not isinstance(q.get('deck'),str) or q.get('status') not in ('qualified','pending','failed'):
            raise ValueError('invalid qualification')
        key=(q['model'],q['deck'])
        if key in qkeys:raise ValueError('duplicate qualification')
        qkeys.add(key)
    if len(data.get('qualifications',[]))>4096 or len(data.get('events',[]))>200:raise ValueError('too many records')
    for event in data.get('events',[]):
        if not isinstance(event,dict):raise ValueError('event object required')
    sampling=data.get('opponent_sampling',{})
    if not isinstance(sampling,dict):raise ValueError('sampling object required')
    for kind in ('planned','actual'):
        values=sampling.get(kind,{})
        if not isinstance(values,dict):raise ValueError('sampling values object required')
        for key,value in values.items():
            if key not in ('latest','history','base') or isinstance(value,bool) or not isinstance(value,(int,float)) or value<0 or (kind=='planned' and value>1):
                raise ValueError('invalid sampling values')
    if not isinstance(data.get('telemetry',{}),dict):raise ValueError('telemetry object required')
    for field in ('generated_at','heartbeat_at'):
        if data.get(field):
            if not isinstance(data[field],str):raise ValueError('timestamp string required')
            stamp=datetime.fromisoformat(data[field].replace('Z','+00:00'))
            if stamp.tzinfo is None:raise ValueError('timezone required')
    return data


def write_snapshot(path, data):
    """Producer integration point: atomic metadata publication, not a training control."""
    validate(data)
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(encoded)>MAX_BYTES:raise ValueError('snapshot too large')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def empty_state():
    return dict(schema_version=1,run_name='BC → 自博弈联赛',status='not_connected',
        evaluation_id=None,models=[dict(id='bc-1037042',name='BC · 1037042',role='base',parent=None,
        elo=None,history=[],parameters=4804205)],matchups=[],qualifications=[],events=[],
        note='当前只确认基模身份；新模型联赛评估尚未接入。压测吞吐不能换算为 Elo 或实力。')


def demo_state():
    """Fictional deterministic display fixture, never persisted into a real source."""
    names=[('bc','BC 基模','base',None,1000),('p1','P001 · 稳健','historical','bc',1090),
           ('p2','P002 · 快攻','historical','p1',1160),('p3','P003 · 防守','champion','p1',1220),
           ('p4','P004 · 反制','candidate','p3',1250),('x1','X001 · 克制探索','exploiter','p2',1140)]
    models=[]
    for i,(key,name,role,parent,elo) in enumerate(names):
        models.append(dict(id=key,name=name,role=role,parent=parent,elo=elo,
            history=[dict(step=t*10000,elo=1000 if i==0 else 1000+(elo-1000)*t/8+(0 if t==8 else ((t+i)%3-1)*12)) for t in range(9)],
            deploys_per_minute=7+i*.8,defense_response_seconds=2.6-i*.22))
    matches=[]
    for i,a in enumerate(models):
        for j,b in enumerate(models):
            if i>=j:continue
            for d,deck in enumerate(('野猪循环','矿工毒药','巨人推进')):
                wins=max(2,min(38,round(20+(a['elo']-b['elo'])/20)+((i+j+d)%3-1)*5))
                if (i,j) in ((2,3),(3,4)):wins=29+d
                if (i,j)==(2,4):wins=9+d
                if (i,j)==(3,5):wins=9+d
                matches.append(dict(a=a['id'],b=b['id'],deck=deck,scope='evaluation',wins=wins,draws=2,losses=40-wins))
    return dict(schema_version=1,run_name='联赛演示 · 合成数据',status='demo',evaluation_id='DEMO-ANCHOR-01',
        generated_at='2026-09-11T00:00:00Z',models=models,matchups=matches,
        qualifications=[dict(model=m['id'],deck=d,status=('pending' if m['id']=='p4' and d!='野猪循环' else 'qualified'))
                        for m in models for d in ('野猪循环','矿工毒药','巨人推进')],
        opponent_sampling=dict(planned=dict(latest=.4,history=.4,base=.2),actual=dict(latest=240,history=330,base=430)),
        telemetry=dict(unique_learner_samples_per_second=860,queue_seconds=3.2,discarded_long_games=0),
        events=[dict(time='演示',type='评估',text='P002 → P003 → P004 → P002 出现循环克制，不能只看总评级。'),
                dict(time='演示',type='资质',text='P004 的两套非固定卡组尚未通过资质检查。')],
        note='所有 Elo、对战、行为和训练数据均为合成样例，不代表当前项目表现。')


class SnapshotReader:
    def __init__(self,path=None):self.path=Path(path) if path else None;self.key=None;self.value=None;self.lock=threading.Lock()
    def read(self):
        if self.path is None:return empty_state()
        with self.lock:
            stat=self.path.stat()
            if stat.st_size>MAX_BYTES:raise ValueError('snapshot too large')
            key=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
            if key!=self.key:
                content=self.path.read_bytes()
                if len(content)>MAX_BYTES:raise ValueError('snapshot too large')
                value=json.loads(content.decode('utf-8-sig'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('non-finite JSON')))
                self.value=validate(value);self.key=key
            return self.value


def make_server(port=19731,snapshot=None):
    reader=SnapshotReader(snapshot);html=Path(__file__).with_name('league_dashboard.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json; charset=utf-8'):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:return self.send(403,b'{"error":"invalid host"}')
            url=urlsplit(self.path)
            if url.path=='/':return self.send(200,html,'text/html; charset=utf-8')
            if url.path!='/api/state':return self.send(404,b'{"error":"not found"}')
            try:
                demo=parse_qs(url.query).get('demo')==['1']
                payload=dict(mode='demo' if demo else 'real',data=demo_state() if demo else reader.read(),
                             served_at=datetime.now(timezone.utc).isoformat())
                return self.send(200,json.dumps(payload,ensure_ascii=False,allow_nan=False).encode())
            except (OSError,ValueError,KeyError,TypeError,RecursionError):
                return self.send(503,json.dumps(dict(error='快照缺失、写入中或格式无效；不展示旧数据冒充实时。'),ensure_ascii=False).encode())
        def do_POST(self):self.send(405,b'{"error":"read-only monitor"}')
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=19731);parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args();server=make_server(args.port,args.snapshot)
    print(f'League monitor: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
