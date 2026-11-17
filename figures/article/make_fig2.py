from pathlib import Path
import os,sys,csv,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper-figure2-style-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).parent.resolve();sys.path.insert(0,str(D.parents[2]/'benchmark/scripts'))
from _plot_style import set_publication_style
set_publication_style();plt.rcParams['svg.fonttype']='none';plt.rcParams['pdf.fonttype']=42
rows=list(csv.DictReader((D/'Fig_2_data.csv').open()))
axes=['completeness','safety','sop_applicability','no_uncertainty','hallucination_severity'];labels=['Полнота','Медицинская\nбезопасность','Применимость\nСОП','Признание\nнеопределённости','Тяжесть\nгаллюцинаций']
cats=[('expert1','direct','Эксперт 1, closed-book','#5471aa'),('expert1','rag','Эксперт 1, RAG','#d0885c'),('expert2','direct','Эксперт 2, closed-book','#7e94bd'),('expert2','rag','Эксперт 2, RAG','#dbab8d'),('judge','direct','LLM-судьи, closed-book','#b8c6df'),('judge','rag','LLM-судьи, RAG','#ecd1bf')]
fig,ax=plt.subplots(figsize=(9.5,8.5),layout='constrained');y=np.arange(5);h=.125;plotted=[]
for i,(source,mode,label,color) in enumerate(cats):
 a=[next(r for r in rows if r['axis']==k and r['source']==source and r['mode']==mode) for k in axes];plotted+=a
 v=np.array([float(r['mean']) for r in a]);lo=np.array([float(r['ci_low']) for r in a]);hi=np.array([float(r['ci_high']) for r in a]);pos=y+(i-2.5)*h
 ax.barh(pos,v,h,label=label,color=color,edgecolor='black',linewidth=.7,xerr=[v-lo,hi-v],error_kw={'ecolor':'black','elinewidth':.9,'capsize':2.5})
 for yy,m,l,u in zip(pos,v,lo,hi):ax.text(u+.03,yy,f'{m:.2f} [{l:.2f}–{u:.2f}]',va='center',fontsize=7.5)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,3.65);ax.set_xticks([0,1,2,3]);ax.set_xlabel('Средняя оценка по шкале критерия')
ax.grid(True,axis='x',alpha=.25,linestyle='--');ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),borderaxespad=0,frameon=False,fontsize=10)
for ext in ['svg']:fig.savefig(D/('Fig_2.'+ext),dpi=300)
assert len(plotted)==30
print('PASS: 30 series/criterion rows, equal figure masks')
