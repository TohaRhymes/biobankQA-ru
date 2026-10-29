"""Exploratory revision analysis. Six models fixed; task is resampling unit."""
import argparse,collections,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
AXES=['completeness','safety','sop_applicability','no_uncertainty','hallucination_severity','use_in_work']
def bh(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q.tolist()
def consortium(r):
 out=collections.defaultdict(list)
 for j,scores in (r.get('open_scores_individual') or {}).items():
  if j==r['provider']+':'+r['model']:continue
  for a,v in (scores or {}).items():
   if isinstance(v,(int,float)):out[a].append(v)
 return {a:float(np.mean(v)) for a,v in out.items()}
def paired_tasks(rows):
 pairs=collections.defaultdict(dict)
 for r in rows:
  k=(r['item_id'],r['provider'],r['model']);m=r['mode']
  if m in pairs[k]:raise ValueError('duplicate pair')
  pairs[k][m]=r['score']
 tasks=collections.defaultdict(list);n=0
 for (t,_,_),p in sorted(pairs.items()):
  if set(p)>={'direct','rag'}:tasks[t].append([p['direct'],p['rag']]);n+=1
 return np.array([np.mean(v,axis=0) for _,v in sorted(tasks.items())]),n
def exact_signflip(d):
 d=np.asarray(d);signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
 return float(np.mean(np.abs(signs@d)>=abs(d.sum())-1e-12))
def bootstrap(d):
 d=np.asarray(d);rng=np.random.default_rng(20261018);v=d[rng.integers(len(d),size=(20000,len(d)))].mean(axis=1);return np.quantile(v,[.025,.975]).tolist()
def old_wilcoxon(d):
 d=np.asarray(d);d=d[np.abs(d)>1e-12]
 return float(wilcoxon(d,method='auto').pvalue) if len(d) else 1.
def load(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save_csv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
