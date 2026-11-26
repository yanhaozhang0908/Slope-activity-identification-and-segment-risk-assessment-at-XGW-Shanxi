"""Label-free neighboring-weight robustness followed by exploratory label selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.stats import rankdata
from scipy.special import xlogy
from experiment_insar_weights import BASE, read
from evaluate_protection_control_risk import ROOT, VAL
from experiment_scale_aggregation_validation import fit, design, standard, grouped

OUT=ROOT/'outputs/weight_search_20260909'
ZONES=['protection','control']
RULES={'step':.05,'minimum_component':.10,'minimum_neighbor_spearman':.90,
       'minimum_neighbor_top20_retention':.80,'near_best_mean_deviance_delta':2,
       'selection':'Pass robustness in both zones; positive Poisson slopes in both; mean deviance within 2 of eligible best; maximize worst-zone neighbor top20 retention, then worst-zone rank correlation, then mean deviance. Capture fraction is descriptive, not another tuning criterion.',
       'inference':'No cross-validation and no untouched external test. No significance claims after selection.'}

def save(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(RULES,ensure_ascii=False,indent=2),encoding='utf-8')
    gdal.UseExceptions()
    ds=gdal.Open(str(BASE/'components_10m.tif'))
    comp={ds.GetRasterBand(i).GetDescription():ds.GetRasterBand(i).ReadAsArray().astype(float) for i in range(1,ds.RasterCount+1)}
    owner,_=read(BASE/'nearest_segment_id_10m.tif')
    old=pd.read_csv(BASE/'segment_risk.csv')
    weights=np.array([(a,b,20-a-b) for a in range(2,17) for b in range(2,19-a) if 20-a-b>=2])/20
    n=len(weights); scores={}; frames={}; rows=[]
    for zone in ZONES:
        mask=np.isfinite(read(BASE/f'{zone}_risk_score_10m.tif')[0])
        d=old[old.zone.eq(zone)].sort_values('start_m').reset_index(drop=True)
        arr=np.empty((n,len(d)))
        for j,r in enumerate(d.itertuples()):
            sel=mask&(owner==r.segment_id)
            c=np.stack([comp[k][sel] for k in ['environment','insar','landslide']])
            assert c.shape[1]>0 and np.isfinite(c).all() and c.min()>=0
            v=100*weights@c
            arr[:,j]=.7*v.mean(axis=1)+.3*np.percentile(v,90,axis=1)
        scores[zone]=arr;frames[zone]=d
        for i,w in enumerate(weights):
            for j,r in enumerate(d.itertuples()):
                rows.append(dict(candidate=i,zone=zone,segment_id=r.segment_id,start_m=r.start_m,length_m=r.length_m,risk_score=arr[i,j]))
    save('all_segment_scores.csv',rows)
    # This stage has no access to instability labels.
    internal=[]
    for i,w in enumerate(weights):
        neighbor=np.flatnonzero(np.isclose(np.abs(weights-w).sum(axis=1),.1)&(np.max(np.abs(weights-w),axis=1)<.051))
        for zone in ZONES:
            a=scores[zone];top=set(np.argsort(-a[i],kind='stable')[:32])
            rho=[np.corrcoef(rankdata(a[i]),rankdata(a[k]))[0,1] for k in neighbor]
            retention=[len(top&set(np.argsort(-a[k],kind='stable')[:32]))/32 for k in neighbor]
            internal.append(dict(candidate=i,zone=zone,min_rho=min(rho),min_retention=min(retention),neighbors=len(neighbor)))
    save('internal_robustness.csv',internal)
    agg=pd.DataFrame(internal).groupby('candidate')[['min_rho','min_retention']].min()
    agg['internal_pass']=(agg.min_rho>=.90)&(agg.min_retention>=.80)
    # Labels are first read after the internal assessment is complete.
    cov=pd.read_csv(VAL/'validation_segments.csv').set_index('segment_id').coverage
    counts=pd.read_csv(VAL/'instability_object_unique_segment_assignment_50m.csv').groupby('segment_id').size()
    external=[];validation={}
    for zone in ZONES:
        d=frames[zone].copy();d['count']=d.segment_id.map(counts).fillna(0).astype(int)
        select=d.segment_id.map(cov).eq(1).to_numpy();d=d[select].copy()
        assert len(d)==90 and d['count'].sum()==70
        validation[zone]=(d,select)
        y=d['count'].to_numpy();e=d.length_m.to_numpy()/1000
        for i in range(n):
            x=scores[zone][i,select];m=fit(design(standard(x)),y,e)
            dev=2*np.sum(xlogy(y,y/m['mu'])-(y-m['mu']))
            external.append(dict(candidate=i,zone=zone,deviance=dev,irr=np.exp(m['beta'][1]),
                top20_capture=y[np.argsort(-x,kind='stable')[:18]].sum()/y.sum()))
    save('external_association.csv',external)
    ext=pd.DataFrame(external)
    tab=agg.join(ext.groupby('candidate').agg(mean_deviance=('deviance','mean'),min_irr=('irr','min'),mean_capture=('top20_capture','mean')))
    for j,k in enumerate(['w_environment','w_insar','w_landslide']):tab[k]=weights[:,j]
    eligible=tab.internal_pass&(tab.min_irr>1)
    assert eligible.any(),'No candidate passes prespecified criteria'
    best=tab.loc[eligible,'mean_deviance'].min()
    tab['near_best']=eligible&(tab.mean_deviance<=best+2)
    pool=tab[tab.near_best].sort_values(['min_retention','min_rho','mean_deviance'],ascending=[False,False,True])
    chosen=int(pool.index[0]);tab['selected']=tab.index==chosen
    save('candidate_comparison.csv',tab.reset_index());save('acceptable_weights.csv',pool.reset_index())
    selected=pd.DataFrame(rows);selected=selected[selected.candidate.eq(chosen)].copy()
    selected['grade']=np.minimum((selected.risk_score//20).astype(int)+1,5)
    save('selected_segment_scores.csv',selected)
    groups=[];classes=[]
    for zone in ZONES:
        d,sel=validation[zone];d=d.copy();d['risk_score']=scores[zone][chosen,sel]
        d['quintile']=pd.qcut(d.risk_score.rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
        groups.extend(dict(zone=zone,**r) for r in grouped(d,'quintile'))
        for grade in range(1,6):
            t=selected[(selected.zone==zone)&(selected.grade==grade)]
            classes.append(dict(zone=zone,grade=grade,segments=len(t),length_km=t.length_m.sum()/1000))
    save('selected_quintile_density.csv',groups);save('selected_class_lengths.csv',classes)
    plt.rcParams.update({'font.family':['Times New Roman','SimSun'],'font.size':9,'axes.unicode_minus':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(2,2,figsize=(7.4,6.0),layout='constrained')
    sc=ax[0,0].scatter(tab.w_environment,tab.w_insar,c=tab.mean_deviance,cmap='viridis_r',s=24)
    ax[0,0].scatter(pool.w_environment,pool.w_insar,facecolors='none',edgecolors='orange',s=50)
    ax[0,0].scatter(*weights[chosen,:2],marker='*',s=130,c='red')
    ax[0,0].set(xlabel='环境权重',ylabel='InSAR权重',title='(a) 候选权重与拟合偏差')
    fig.colorbar(sc,ax=ax[0,0],shrink=.85)
    ax[0,1].scatter(tab.min_retention,tab.mean_deviance,s=20,color='#888888')
    ax[0,1].scatter(tab.loc[chosen,'min_retention'],tab.loc[chosen,'mean_deviance'],marker='*',s=130,color='red')
    ax[0,1].set(xlabel='前20%分段最低保留率',ylabel='平均Poisson偏差',title='(b) 稳健性与外部对应性')
    for zone,color,label in zip(ZONES,['#0072B2','#D55E00'],['保护范围','建设控制地带']):
        t=selected[selected.zone==zone];ax[1,0].plot((t.start_m+t.length_m/2)/1000,t.risk_score,lw=.8,color=color,label=label)
        q=pd.DataFrame(groups);q=q[q.zone==zone]
        ax[1,1].plot(q.group,q.density,'o-',color=color,label=label)
    ax[1,0].set(xlabel='沿长城里程（km）',ylabel='风险指数',ylim=(0,100),title='(c) 所选权重沿程风险')
    ax[1,1].set(xlabel='等数量得分组（非固定风险等级）',ylabel='识别失稳对象密度（个/km）',xticks=[1,2,3,4,5],title='(d) 结构失稳对应性（探索性）')
    ax[1,0].legend(frameon=False);ax[1,1].legend(frameon=False)
    for a in ax.flat:a.spines[['top','right']].set_visible(False)
    for suffix in ['png','svg','tiff']:fig.savefig(OUT/f'weight_selection.{suffix}',dpi=600)
    plt.close(fig)
    summary={'candidate_count':n,'internal_pass':int(tab.internal_pass.sum()),'near_best_count':len(pool),'selected_candidate':chosen,'weights_E_V_L':weights[chosen].tolist(),'selected_metrics':tab.loc[chosen].to_dict(),'zones':ext[ext.candidate.eq(chosen)].to_dict('records'),'rules':RULES}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(pool.to_string());print(pd.DataFrame(groups).to_string(index=False))

if __name__=='__main__':main()
