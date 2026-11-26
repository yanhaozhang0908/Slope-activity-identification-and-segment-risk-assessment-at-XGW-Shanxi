"""Aggregate original signed displacement to fixed common-cell slope support."""
import compare_response_statistics as exp
import numpy as np
import pandas as pd
import geopandas as gpd
import json,time
ROOT=exp.ROOT;OUT=ROOT/'outputs/scientific_prediction_upgrade_20260914';OUT.mkdir(parents=True,exist_ok=True)

def main():
    ref=exp.ref.base.old.base
    support=pd.read_pickle(ROOT/'outputs/full_slope_spatial_gam_20260911/observed_pixels.pkl')
    support=support[support.target.notna()].set_index('cell')
    u=gpd.read_file(exp.OUT/'slope_targets.gpkg',layer='targets');ids=u[u.eligible.eq(1)].SU_ID.to_numpy()
    schema=json.loads((OUT/'insar_schema_audit.json').read_text(encoding='utf-8'));allrows=[];audit=[]
    for track,path in ref.TRACKS.items():
        dest=OUT/f'{track}_slope_timeseries.csv'
        if dest.exists():print('Cached',track,flush=True);continue
        fields=next(x['fields'] for x in schema if x['path'].replace('\\','/')==str(path).replace('\\','/'))
        dates=[f for f in fields if f.startswith('D_')];start=time.time()
        print('Read',track,len(dates),'epochs',flush=True)
        p=gpd.read_file(path,columns=['velocity','coherence','Vprecision','ALOS','ILOS']+dates+['geometry']).to_crs(u.crs)
        p=p[p.coherence.ge(.65)&p.velocity.notna()&p.geometry.notna()&~p.geometry.is_empty].copy()
        cc=np.floor((p.geometry.x-ref.GT[0])/ref.GT[1]).astype(int);rr=np.floor((p.geometry.y-ref.GT[3])/ref.GT[5]).astype(int)
        p['cell']=rr*ref.WIDTH+cc
        p=p[p.cell.isin(support.index)].copy();p=p.drop(columns='geometry')
        cells=p.groupby('cell')[dates+['velocity','Vprecision','ALOS','ILOS']].median()
        cells['SU_ID']=support.SU_ID.reindex(cells.index)
        err=np.max(np.abs(cells.velocity-support[track].reindex(cells.index)))
        assert err<1e-8 and cells.SU_ID.notna().all()
        ts=cells.groupby('SU_ID')[dates].mean().reindex(u.SU_ID)
        assert ts.loc[ids].notna().all().all()
        rows=[]
        for f in dates:
            rows.extend([dict(track=track,SU_ID=int(uid),date=f[2:6]+'-'+f[6:8]+'-'+f[8:10],displacement=float(v)) for uid,v in ts[f].items()])
        pd.DataFrame(rows).to_csv(dest,index=False,encoding='utf-8-sig')
        quality=cells.groupby('SU_ID')[['velocity','Vprecision','ALOS','ILOS']].median().reset_index();quality.to_csv(OUT/f'{track}_observation_metadata.csv',index=False,encoding='utf-8-sig')
        audit.append(dict(track=track,n_epochs=len(dates),n_cells=len(cells),n_points=len(p),velocity_reproduction_max_error=float(err),dates=dates,seconds=time.time()-start))
        print('AGGREGATED',track,'epochs',len(dates),'seconds',round(time.time()-start,1),flush=True)
    (OUT/'timeseries_preparation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
