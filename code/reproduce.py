"""Offline reproduction from pseudonymized, allowlisted release inputs. No API calls."""
import argparse,collections,json,csv,os,math,random
from pathlib import Path
import sys

if __name__ == '__main__':
 missing = [name for name in ('tasks.jsonl', 'candidate_answers.jsonl', 'judge_scores.jsonl', 'annotations.jsonl', 'iteration_diagnostics.jsonl') if not (Path(__file__).resolve().parents[1] / 'data' / name).is_file()]
 if missing:
  sys.exit('Full reproduction requires withheld evaluation inputs. This reduced release includes saved tables and figures in results/reproduced/. Run full-data analyses in a separate private checkout.')

import numpy as np
from scipy.stats import pearsonr,spearmanr,binomtest
from core import AXES,paired_tasks,bootstrap,exact_signflip,bh,save_csv,consortium,load,old_wilcoxon
from judge_audit import compute_self_preference,compute_loo_rankings
ROOT=Path(__file__).resolve().parents[1]
def kappa(x,y,mx):
 x=np.asarray(x);y=np.asarray(y);ex=sum(np.mean(x==i)*np.mean(y==j)*(i-j)**2 for i in range(mx+1) for j in range(mx+1));return float(1-np.mean((x-y)**2)/ex) if ex else None

def calculate(out):
 out.mkdir(parents=True,exist_ok=True);h=load(ROOT/'data/annotations.jsonl');allr=load(ROOT/'data/candidate_answers.jsonl');judger=load(ROOT/'data/judge_scores.jsonl');j={(r['item_id'],r['provider'],r['model'],r['mode']):consortium(r) for r in judger}
 effects=[];corr=[];iaa=[];rates=[]
 for who in ['expert_1','expert_2']:
  rows=[]
  for axis in AXES:
   rr=[r for r in h if r['annotator']==who and r['axis']==axis];arr,n=paired_tasks(rr);d=arr[:,1]-arr[:,0];lo,hi=bootstrap(d);rows.append(dict(annotator=who,axis=axis,n_pairs=n,n_tasks=len(d),direct_task_mean=float(arr[:,0].mean()),rag_task_mean=float(arr[:,1].mean()),delta_task_mean=float(d.mean()),ci_low=lo,ci_high=hi,p_exact_task=exact_signflip(d)))
   xx=[r['score'] for r in rr];yy=[j[(r['item_id'],r['provider'],r['model'],r['mode'])][axis] for r in rr];corr.append(dict(annotator=who,axis=axis,n=len(xx),human_mean=float(np.mean(xx)),judge_mean=float(np.mean(yy)),pearson=float(pearsonr(xx,yy).statistic),spearman=float(spearmanr(xx,yy).statistic)))
  for r,q in zip(rows,bh([r['p_exact_task'] for r in rows])):r['p_bh_six']=q
  effects+=rows
 for r,q in zip(effects,bh([r['p_exact_task'] for r in effects])):r['p_bh_twelve_sensitivity']=q
 for axis in AXES:
  paired=collections.defaultdict(dict)
  for r in h:
   if r['axis']==axis:paired[(r['item_id'],r['provider'],r['model'],r['mode'])][r['annotator']]=r['score']
  vv=[v for v in paired.values() if len(v)==2];x=[v['expert_1'] for v in vv];y=[v['expert_2'] for v in vv];mx=2 if axis=='hallucination_severity' else 1 if axis=='use_in_work' else 3
  iaa.append(dict(axis=axis,n_pairs=len(x),n_tasks=14,mean_expert_1=float(np.mean(x)),mean_expert_2=float(np.mean(y)),agreement=float(np.mean(np.array(x)==np.array(y))),quadratic_kappa=kappa(x,y,mx),spearman=float(spearmanr(x,y).statistic)))
 grouped=collections.defaultdict(list)
 for r in h:
  if r['axis']=='use_in_work':grouped[(r['annotator'],r['provider'],r['model'],r['mode'])].append(r['score'])
 for (who,p,m,mode),v in sorted(grouped.items()):rates.append(dict(annotator=who,provider=p,model=m,mode=mode,n=len(v),yes=int(sum(v)),fraction=float(np.mean(v))))
 save_csv(out/'effects-final.csv',effects);save_csv(out/'iaa-final.csv',iaa);save_csv(out/'judge-correlations-final.csv',corr);save_csv(out/'usework-by-model.csv',rates)
 # MCQ denominator: valid returned responses; paired effects use common items only.
 mcq=[];sens=[];tasks={r['id']:r for r in load(ROOT/'data/tasks.jsonl')};scoring_checks=0
 for r in allr:
  if r['item_type']=='mcq' and r.get('mcq_score') is not None:
   gold=set(tasks[r['item_id']]['gold_letters_canonical']);pred=set(r['mcq_letters_canonical']);f=2*len(gold&pred)/(len(gold)+len(pred));assert abs(round(f,3)-r['mcq_score'])<1e-12;assert (gold==pred)==bool(r['mcq_strict_correct']);scoring_checks+=1
 for prov,model in sorted({(r['provider'],r['model']) for r in allr if r['item_type']=='mcq'}):
  rows=[r for r in allr if r['item_type']=='mcq' and (r['provider'],r['model'])==(prov,model)];valid=[dict(r,score=r['mcq_score']) for r in rows if r.get('mcq_score') is not None and not r['error']];arr,n=paired_tasks(valid);d=np.round(arr[:,1]-arr[:,0],12);lo,hi=bootstrap(d);modes={m:[r['mcq_score'] for r in valid if r['mode']==m] for m in ['direct','rag']};em={m:[int(bool(r.get('mcq_strict_correct'))) for r in valid if r['mode']==m] for m in ['direct','rag']}
  mcq.append(dict(direct_exact_match=float(np.mean(em['direct'])),rag_exact_match=float(np.mean(em['rag'])),direct_exact_correct=sum(em['direct']),rag_exact_correct=sum(em['rag']),provider=prov,model=model,n_pairs=n,direct_full_n=len(modes['direct']),rag_full_n=len(modes['rag']),direct_full_mean=float(np.mean(modes['direct'])),rag_full_mean=float(np.mean(modes['rag'])),direct_paired_mean=arr[:,0].mean(),rag_paired_mean=arr[:,1].mean(),delta=d.mean(),ci_low=lo,ci_high=hi,p_wilcoxon=old_wilcoxon(d),p_unrounded=old_wilcoxon(arr[:,1]-arr[:,0])))
  arr,n=paired_tasks([dict(r,score=r['mcq_score'] if r.get('mcq_score') is not None else 0.) for r in rows if not r['error']]);d=np.round(arr[:,1]-arr[:,0],12);lo,hi=bootstrap(d);sens.append(dict(provider=prov,model=model,n_pairs=n,direct_mean=float(arr[:,0].mean()),rag_mean=float(arr[:,1].mean()),delta=float(d.mean()),ci_low=lo,ci_high=hi,p=old_wilcoxon(d)))
 for r,q in zip(mcq,bh([r['p_wilcoxon'] for r in mcq])):r['p_bh_twelve']=q
 for r,q in zip(sens,bh([r['p'] for r in sens])):r['p_bh_twelve']=q
 save_csv(out/'mcq.csv',mcq);save_csv(out/'mcq_invalid_zero_sensitivity.csv',sens)
 # Primary-expert figure 2 and answer-level Table 5, common complete pairs.
 fig=[];table5=[]
 for axis in AXES:
  valid={(r['item_id'],r['provider'],r['model'],r['mode']):r['score'] for r in h if r['axis']==axis and r['annotator']=='expert_1'};common={k:v for k,v in valid.items() if axis in j[k]};x=list(common.values());y=[j[k][axis] for k in common];table5.append(dict(axis=axis,n_answers=len(x),expert_answer_mean=float(np.mean(x)),judge_answer_mean=float(np.mean(y)),pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic)))
  pairs={k:v for k,v in common.items() if (*k[:3],'rag' if k[3]=='direct' else 'direct') in common}
  for source in ['expert','llm']:
   for mode in ['direct','rag']:
    t=collections.defaultdict(list)
    for k,v in pairs.items():
     if k[3]==mode:t[k[0]].append(v if source=='expert' else j[k][axis])
    means=np.array([np.mean(v) for _,v in sorted(t.items())]);lo,hi=bootstrap(means);fig.append(dict(axis=axis,source=source,mode=mode,mean=float(means.mean()),ci_low=lo,ci_high=hi,n_tasks=len(t),n_answers=sum(len(v) for v in t.values())))
 save_csv(out/'figure.csv',fig);save_csv(out/'table5_correlations.csv',table5)
 selfrows,judges,_=compute_self_preference(judger);loo=compute_loo_rankings(judger,judges);order=[judges[0],judges[3],judges[1],judges[2]]
 tab6=[dict(judge=z,values=[np.mean([r['mean_gap_vs_others'] for r in selfrows if r['judge']==z and r['axis']==a and r['match_type']==t]) for a in ['safety','no_uncertainty'] for t in ['self','unrelated']]) for z in order];tab7=[dict(axis=a,values=[loo[a]['loo'][z]['spearman'] for z in judges]) for a in AXES[:5]]
 (out/'judge_audit.json').write_text(json.dumps(dict(table6=tab6,table7=tab7),indent=2));save_csv(out/'judge_self_preference.csv',selfrows)
 rows=[]
 for key in sorted({(r['provider'],r['model']) for r in judger if r.get('open_scores_individual')}):
  for axis in AXES[:5]:
   rr=[dict(r,score=consortium(r)[axis]) for r in judger if (r['provider'],r['model'])==key and axis in consortium(r)];arr,n=paired_tasks(rr);d=arr[:,1]-arr[:,0];nz=d[np.abs(d)>1e-12];pv=float(binomtest(sum(nz>0),len(nz)).pvalue) if len(nz) else 1;rows.append(dict(model=key[1],axis=axis,n_pairs=n,delta=float(d.mean()),p_sign=pv))
 for r,q in zip(rows,bh([x['p_sign'] for x in rows])):r['p_bh55']=q
 save_csv(out/'appendix_mode_deltas.csv',rows)
 # Same six fixed models, all iterations: diagnostics, not independently validated quality.
 diagnostics=load(ROOT/'data/iteration_diagnostics.jsonl');models={r['model'] for r in h};diag=[]
 for it in ['iter1','iter2','iter3']:
  z=[r for r in diagnostics if r['iteration']==it and r['model'] in models]
  for mode in ['direct','rag']:
   v=[r for r in z if r['mode']==mode];diag.append(dict(iteration=it,mode=mode,n=len(v),mean_length=float(np.mean([r['length'] for r in v if r['has_response']])),phrase1=sum(r['phrase1'] for r in v),phrase2=sum(r['phrase2'] for r in v)))
 save_csv(out/'iteration_diagnostics.csv',diag)
 assert [x['phrase1'] for x in diag]==[0,41,0,0,0,0]
 assert [x['phrase2'] for x in diag]==[0,0,0,33,0,0]
 assert [round(diag[i+1]['mean_length']/diag[i]['mean_length'],2) for i in [0,2,4]]==[.26,.34,1.00]
 # Historical Figure 1 uses 1000 Python-random resamples seed42 and rounded endpoints.
 f1ci=[]
 for rr in mcq:
  for mode in ['direct','rag']:
   values=[x['mcq_score'] for x in allr if x['item_type']=='mcq' and x['model']==rr['model'] and x['mode']==mode and x.get('mcq_score') is not None and not x['error']];rng=random.Random(42);n=len(values);boot=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(1000));f1ci.append(dict(model=rr['model'],mode=mode,n=n,mean=float(np.mean(values)),ci_low=round(boot[25],3),ci_high=round(boot[975],3)))
 save_csv(out/'figure1.csv',f1ci)
 return dict(scoring_checks=scoring_checks,annotations=len(h),models=len(models),effects=effects,figure=fig,mode_deltas=rows,mcq=mcq,iteration_diagnostics=diag)

def verify(out):
 count=0
 for expected in sorted((ROOT/'expected').glob('*.csv')):
  with expected.open() as f:ee=list(csv.DictReader(f))
  with (out/expected.name).open() as f:aa=list(csv.DictReader(f))
  assert len(ee)==len(aa),expected.name
  for rowa,rowe in zip(aa,ee):
   for k,v in rowe.items():
    assert k in rowa,(expected.name,k)
    try:ok=math.isclose(float(v),float(rowa[k]),rel_tol=1e-10,abs_tol=1e-11)
    except (ValueError,TypeError):ok=v==rowa[k]
    assert ok,(expected.name,k,rowa[k],v)
    count+=1
 expected=json.loads((ROOT/'expected/judge_audit.json').read_text());actual=json.loads((out/'judge_audit.json').read_text());assert expected==actual
 return count

def plots(out,r):
 os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 # Reproducible scientific values; layout is intentionally not byte-identical to original Word figures.
 fig,ax=plt.subplots(figsize=(10,6));axes=AXES[:5];ys=np.arange(5)
 for i,(src,mode) in enumerate([('expert','direct'),('expert','rag'),('llm','direct'),('llm','rag')]):
  z=[next(x for x in r['figure'] if x['axis']==a and x['source']==src and x['mode']==mode) for a in axes];v=np.array([x['mean'] for x in z]);ax.barh(ys+i*.18,v,height=.17,label=src+' '+mode,xerr=[v-np.array([x['ci_low'] for x in z]),np.array([x['ci_high'] for x in z])-v])
 ax.set_yticks(ys+.25,axes);ax.legend();fig.tight_layout();fig.savefig(out/'figure2.png',dpi=160);plt.close(fig)
 models=sorted({x['model'] for x in r['mode_deltas']});vals=np.array([[next(x['delta'] for x in r['mode_deltas'] if x['model']==m and x['axis']==a) for a in axes] for m in models]);fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(vals,cmap='coolwarm',vmin=-1,vmax=1);ax.set_xticks(range(5),axes,rotation=20,ha='right');ax.set_yticks(range(len(models)),models);fig.colorbar(im);fig.tight_layout();fig.savefig(out/'figure3.png',dpi=160);plt.close(fig)

def extra_plots(out):
 os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 with (out/'figure1.csv').open() as f:rows=list(csv.DictReader(f))
 models=sorted({r['model'] for r in rows});fig,ax=plt.subplots(figsize=(10,7))
 for i,mode in enumerate(['direct','rag']):
  z=[next(r for r in rows if r['model']==m and r['mode']==mode) for m in models];v=np.array([float(r['mean']) for r in z]);ax.barh(np.arange(len(models))+i*.35,v,height=.3,label=mode,xerr=[v-np.array([float(r['ci_low']) for r in z]),np.array([float(r['ci_high']) for r in z])-v])
 ax.set_yticks(np.arange(len(models))+.15,models);ax.legend();fig.tight_layout();fig.savefig(out/'figure1.png',dpi=160);plt.close(fig)
 with (out/'judge_self_preference.csv').open() as f:rows=list(csv.DictReader(f))
 judges=sorted({r['judge'] for r in rows});fig,axs=plt.subplots(2,1,figsize=(10,8))
 for ax,axis in zip(axs,['completeness','no_uncertainty']):
  for i,kind in enumerate(['self','unrelated']):
   v=[np.mean([float(r['mean_gap_vs_others']) for r in rows if r['judge']==j and r['axis']==axis and r['match_type']==kind]) for j in judges];ax.bar(np.arange(4)+i*.3,v,width=.28,label=kind)
  ax.set_xticks(np.arange(4)+.15,judges,rotation=15,ha='right');ax.set_title(axis);ax.legend()
 fig.tight_layout();fig.savefig(out/'figure4.png',dpi=160);plt.close(fig)
 diagnostics=load(ROOT/'data/iteration_diagnostics.jsonl');models={r['model'] for r in load(ROOT/'data/annotations.jsonl')};fig,axs=plt.subplots(2,3,figsize=(14,8))
 for i,it in enumerate(['iter1','iter2','iter3']):
  zz=[r for r in diagnostics if r['iteration']==it and r['model'] in models];vs=[[r['length'] for r in zz if r['mode']==m and r['has_response']] for m in ['direct','rag']];axs[0,i].violinplot(vs,showmeans=True);axs[0,i].set_xticks([1,2],['direct','rag']);axs[0,i].set_title(it+' length')
  for k,mode in enumerate(['direct','rag']):
   axes=['completeness','safety','actionability','uncertainty_handling','hallucination_severity'];means=[np.mean([r['open_scores'][a] for r in zz if r['mode']==mode and a in r['open_scores']]) for a in axes];axs[1,i].plot(range(5),means,label=mode)
  axs[1,i].set_xticks(range(5),axes,rotation=45,ha='right');axs[1,i].legend()
 fig.tight_layout();fig.savefig(out/'figure5.png',dpi=160);plt.close(fig)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs');parser.add_argument('--no-plots',action='store_true');args=parser.parse_args();result=calculate(args.out);count=verify(args.out)
 if not args.no_plots:plots(args.out,result);extra_plots(args.out)
 (args.out/'validation.json').write_text(json.dumps(dict(status='PASS',checked_csv_cells=count,**{k:result[k] for k in ['scoring_checks','annotations','models']}),indent=2));print('PASS',count,'reference cells;',result['scoring_checks'],'F1/EM recomputations')
