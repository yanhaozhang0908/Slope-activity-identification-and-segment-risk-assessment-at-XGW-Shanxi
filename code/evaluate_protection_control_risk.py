"""Reproducible paired-zone risk experiment; never overwrites the 100 m results."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from osgeo import gdal, ogr
from analyze_landslide_wall_impact import DEM_PATH, FACTOR_PATHS, LANDSLIDES, aligned_array, load_union, raster_grid
from analyze_protection_control_zones import SEGMENTS, build_wall, make_zone_geometries, write_zones
from evaluate_wall_100m_segment_risk import INSAR_TRACKS, load_insar_track, make_target_srs
from evaluate_wall_pixel_risk_100m import (robust_scale_in_mask, rasterize_geometry, pixel_centres,
    local_insar_score, landslide_score, write_float_raster, write_class_raster,
    write_component_raster, classify, COMPONENT_WEIGHTS, ENVIRONMENT_WEIGHTS, CLASS_DEFINITIONS)
from plot_risk_level_instability_density import poisson_irr, poisson_exact_interval

ROOT = Path(r'D:\project\shuozhouinflucefactor')
OUT = ROOT / 'outputs' / 'protection_control_risk_20260907'
VAL = ROOT / 'process' / 'wall_segment_pixel_risk_instability_validation'
KEYS = ['protection', 'control']
LABELS = ['保护范围分析区（0–50 m）', '建设控制地带分析区（50–550 m）']
COLORS = ['#D55E00', '#0072B2']

def save_table(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')

def nearest_segment(xy, segments):
    """Exact point-to-polyline distance; ties go to lowest chainage segment."""
    best = np.full(len(xy), np.inf)
    owner = np.full(len(xy), -1, dtype=int)
    for i, rec in enumerate(segments):
        line = rec['geometry']
        pts = np.array([[line.GetX(j), line.GetY(j)] for j in range(line.GetPointCount())])
        dist = np.full(len(xy), np.inf)
        for a, b in zip(pts[:-1], pts[1:]):
            v = b-a
            t = np.clip((xy-a) @ v / max(float(v @ v), 1e-20), 0, 1)
            d = np.sum((xy-a-t[:, None]*v)**2, axis=1)
            dist = np.minimum(dist, d)
        hit = dist < best-1e-9
        best[hit], owner[hit] = dist[hit], i
    assert np.all(owner >= 0)
    return owner, np.sqrt(best)

def validation(table):
    coverage = pd.read_csv(VAL / 'validation_segments.csv')[['segment_id', 'coverage']]
    assignments = pd.read_csv(VAL / 'instability_object_unique_segment_assignment_50m.csv')
    counts = assignments.groupby('segment_id').size()
    models, groups, all_rows = [], [], []
    for key in KEYS:
        d = table[table.zone.eq(key)].merge(coverage, on='segment_id')
        d = d[d.coverage.eq(1)].sort_values('start_m').copy()
        d['object_count'] = d.segment_id.map(counts).fillna(0).astype(int)
        d['group'] = pd.qcut(d.risk_score.rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
        y, x, exposure = d.object_count.to_numpy(), d.risk_score.to_numpy(), d.length_m.to_numpy()/1000
        fit = poisson_irr(y, x, exposure)
        # Enumerate every shift. Keep finite solutions even when BFGS reports precision loss.
        null = [poisson_irr(np.roll(y, shift), x, exposure)['beta'] for shift in range(1, len(d))]
        assert np.all(np.isfinite(null))
        p = float((1+np.sum(np.array(null) >= fit['beta']))/len(d))
        models.append(dict(zone=key, covered_segments=len(d), objects=int(y.sum()),
                           circular_shift_p_one_sided=p, **fit))
        for group, sub in d.groupby('group'):
            n, km = int(sub.object_count.sum()), float(sub.length_m.sum()/1000)
            lo, hi = poisson_exact_interval(n, km)
            groups.append(dict(zone=key, group=int(group), segment_count=len(sub), objects=n,
                length_km=km, density=n/km, ci_low=lo, ci_high=hi,
                score_min=float(sub.risk_score.min()), score_max=float(sub.risk_score.max())))
        all_rows.extend(d.to_dict('records'))
    # Two related primary association tests: report Holm adjustment alongside raw p.
    order = np.argsort([r['circular_shift_p_one_sided'] for r in models])
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (2-rank)*models[index]['circular_shift_p_one_sided']))
        models[index]['circular_shift_p_holm'] = previous
    save_table('validation_models.csv', models)
    save_table('validation_groups.csv', groups)
    save_table('validation_segments.csv', all_rows)
    return models, pd.DataFrame(groups)

def figures(table, groups, classes, grid):
    plt.rcParams.update({'font.family':['Times New Roman', 'SimSun'], 'axes.unicode_minus':False,
                        'font.size':8, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 1, figsize=(180/25.4, 145/25.4), layout='constrained')
    for key, label, color, style in zip(KEYS, LABELS, COLORS, ['-', '--']):
        d = table[table.zone.eq(key)]
        axes[0].plot((d.start_m+d.end_m)/2000, d.risk_score, style, color=color, lw=1, label=label)
        q = groups[groups.zone.eq(key)]
        offset = -.06 if key == 'protection' else .06
        axes[1].errorbar(q.group+offset, q.density, yerr=[q.density-q.ci_low,q.ci_high-q.density],
                         fmt='o'+style, color=color, lw=1, ms=3, capsize=2, label=label)
    axes[0].set(xlabel='沿长城里程（km）', ylabel='分段风险指数（0–100）', ylim=(0,100), xlim=(0,8))
    axes[1].set(xlabel='风险指数等数量排序组（不是风险等级）', ylabel='本体结构失稳对象密度（个/km）', xticks=[1,2,3,4,5])
    axes[1].set_ylim(0, groups.ci_high.max()*1.23)
    for i, ax in enumerate(axes):
        ax.set_title(['(a) 两区对应分段风险指数', '(b) 与本体结构失稳的关联检验'][i], loc='left')
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','tiff']:
        kw = {'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(OUT / ('Fig1_分区风险与失稳关联.'+ext), dpi=600, **kw)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(180/25.4,170/25.4))
    gt = grid['gt']
    extent=[gt[0],gt[0]+grid['width']*gt[1],gt[3]+grid['height']*gt[5],gt[3]]
    palette = [np.array(c[5][:3])/255 for c in CLASS_DEFINITIONS]
    cmap=ListedColormap(palette)
    rr,cc=np.where((classes['protection']>0)|(classes['control']>0))
    bounds=(gt[0]+cc.min()*gt[1]-300,gt[0]+(cc.max()+1)*gt[1]+300,
            gt[3]+(rr.max()+1)*gt[5]-300,gt[3]+rr.min()*gt[5]+300)
    for ax,key,label in zip(axes, KEYS, LABELS):
        ax.imshow(np.ma.masked_equal(classes[key],0), extent=extent, cmap=cmap,
                  norm=BoundaryNorm([.5,1.5,2.5,3.5,4.5,5.5],5), interpolation='nearest')
        ax.set_title(label, fontsize=8)
        ax.set(xlabel='东坐标（m）', ylabel='北坐标（m）')
        ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.annotate('N', xy=(.9,.96), xytext=(.9,.87), xycoords='axes fraction',
                    ha='center', arrowprops=dict(arrowstyle='->',color='black'))
    fig.legend(handles=[Patch(facecolor=c,label=r[2]) for c,r in zip(palette,CLASS_DEFINITIONS)],
               loc='lower center', ncol=5, frameon=False)
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.17,wspace=.4)
    for ext in ['png','svg','tiff']:
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(OUT / ('Fig2_两区像元风险等级.'+ext),dpi=600,**kw)
    plt.close(fig)

def main():
    gdal.UseExceptions()
    OUT.mkdir(parents=True, exist_ok=True)
    ds=gdal.Open(str(DEM_PATH)); grid=raster_grid(ds); srs=make_target_srs(grid['projection'])
    source=ogr.Open(str(SEGMENTS)); layer=source.GetLayerByName('wall_segment_risk')
    wall=build_wall(layer); zones=make_zone_geometries(wall)
    write_zones(OUT/'analysis_zones.gpkg',zones,srs)
    segments=[]; layer.ResetReading()
    for f in layer:
        segments.append(dict(segment_id=int(f['segment_id']),start_m=float(f['start_m']),
                             end_m=float(f['end_m']),geometry=f.GetGeometryRef().Clone()))
    segments.sort(key=lambda r:r['start_m'])
    masks={k:rasterize_geometry(g,grid) for k,g in zip(KEYS,zones.values())}
    domain=masks['protection']|masks['control']
    assert not np.any(masks['protection'] & masks['control'])
    raw={k:aligned_array(FACTOR_PATHS[p],ds)[0] for k,p in
         [('slope','slope_deg'),('relief','relief_300m'),('twi','twi'),('drainage','distance_to_drainage_m')]}
    cube=aligned_array(FACTOR_PATHS['ndvi'],ds); nd=gdal.Open(str(FACTOR_PATHS['ndvi']))
    growing=[i for i in range(nd.RasterCount) if nd.GetRasterBand(i+1).GetDescription()[-2:] in ['05','06','07','08','09']]
    assert growing, 'Missing NDVI month labels'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning); raw['ndvi']=np.nanmedian(cube[growing],axis=0)
    valid=domain & np.logical_and.reduce([np.isfinite(a) for a in raw.values()])
    scaling={}; components={}
    for k,a in raw.items():
        components[k],scaling[k]=robust_scale_in_mask(a,valid,invert=k in ['drainage','ndvi'])
    components['environment']=sum(ENVIRONMENT_WEIGHTS[k]*components[k] for k in ENVIRONMENT_WEIGHTS)
    rows,cols,xy=pixel_centres(valid,grid['gt'])
    print('Factors ready; calculating InSAR and landslide fields',flush=True)
    tracks={name:load_insar_track(p,srs) for name,p in INSAR_TRACKS.items()}
    components['insar'],insar_stats=local_insar_score(tracks,xy,valid.shape,rows,cols)
    observed=np.logical_or.reduce([t['tree'].query(xy)[0]<=100 for t in tracks.values()])
    missing=np.zeros(valid.shape,bool); missing[rows,cols]=~observed
    slides={name:load_union(p,srs) for name,p in LANDSLIDES.items()}
    components['landslide']=landslide_score(slides,xy,valid.shape,rows,cols)
    risk=100*sum(COMPONENT_WEIGHTS[k]*components[k] for k in COMPONENT_WEIGHTS)
    # Missing InSAR is not evidence for no deformation. Exclude it from primary results.
    risk[missing]=np.nan
    write_component_raster(OUT/'components_10m.tif',list(components.values()),list(components),ds)
    print('Assigning pixels uniquely to nearest wall segment',flush=True)
    owner,distance=nearest_segment(xy,segments)
    ownership=np.full(valid.shape,np.nan); ownership[rows,cols]=np.array([r['segment_id'] for r in segments])[owner]
    write_float_raster(OUT/'nearest_segment_id_10m.tif',ownership,ds,'Unique nearest segment ID')
    records=[]; zone_summary=[]; area_rows=[]; length_rows=[]; class_arrays={}
    for key in KEYS:
        mask=masks[key]&valid&~missing
        array=np.where(mask,risk,np.nan)
        cl=classify(array,mask); class_arrays[key]=cl
        write_float_raster(OUT/f'{key}_risk_score_10m.tif',array,ds,'Relative risk index 0-100')
        write_class_raster(OUT/f'{key}_risk_class_10m.tif',cl,ds)
        vals=array[mask]
        zone_summary.append(dict(zone=key,pixel_count=len(vals),area_km2=len(vals)*.0001,
            mean=float(vals.mean()),p90=float(np.percentile(vals,90)),minimum=float(vals.min()),maximum=float(vals.max()),
            missing_factor_pixels=int(np.sum(masks[key]&~valid)),missing_insar_pixels=int(np.sum(masks[key]&missing))))
        for i,rec in enumerate(segments):
            selection=mask & (ownership==rec['segment_id']); v=risk[selection]
            assert len(v)>0, (key,rec['segment_id'])
            avg,p90=float(v.mean()),float(np.percentile(v,90)); score=.7*avg+.3*p90
            code=min(int(score//20)+1,5)
            # Sensitivity: original overlapping segment-buffer approach, clipped to global zones.
            local=rasterize_geometry(rec['geometry'].Buffer(50 if key=='protection' else 550,32),grid)&mask
            lv=risk[local]
            overlap_score=.7*float(lv.mean())+.3*float(np.percentile(lv,90))
            records.append(dict(zone=key,segment_id=rec['segment_id'],start_m=rec['start_m'],end_m=rec['end_m'],
                length_m=rec['end_m']-rec['start_m'],pixel_count=len(v),pixel_sum=float(v.astype(float).sum()),
                pixel_mean=avg,pixel_p90=p90,risk_score=score,class_code=code,class_cn=CLASS_DEFINITIONS[code-1][2],
                overlap_sensitivity_score=overlap_score))
        for code,_,cn,_,_,_ in CLASS_DEFINITIONS:
            count=int(np.sum(cl==code))
            area_rows.append(dict(zone=key,class_code=code,class_cn=cn,pixels=count,area_km2=count*.0001,percent=100*count/len(vals)))
    table=pd.DataFrame(records)
    for key in KEYS:
        d=table[table.zone.eq(key)]
        for code,_,cn,_,_,_ in CLASS_DEFINITIONS:
            q=d[d.class_code.eq(code)]
            length_rows.append(dict(zone=key,class_code=code,class_cn=cn,segment_count=len(q),length_km=q.length_m.sum()/1000,
                                   length_percent=100*q.length_m.sum()/d.length_m.sum()))
        assert d.pixel_count.sum()==next(r['pixel_count'] for r in zone_summary if r['zone']==key)
    save_table('segment_risk.csv',records); save_table('zone_summary.csv',zone_summary)
    save_table('pixel_class_area.csv',area_rows); save_table('segment_class_length.csv',length_rows)
    # Global pixel means and block-bootstrap CI use exactly the same area-weighted estimand.
    rng=np.random.default_rng(20260907); blocks=np.array_split(np.arange(len(segments)),8)
    p=table[table.zone.eq('protection')]; c=table[table.zone.eq('control')]
    ratios=[]
    for _ in range(5000):
        ix=np.concatenate([blocks[b] for b in rng.integers(0,8,8)])
        a=p.iloc[ix]; b=c.iloc[ix]
        ratios.append((a.pixel_sum.sum()/a.pixel_count.sum())/(b.pixel_sum.sum()/b.pixel_count.sum()))
    ci=np.percentile(ratios,[2.5,97.5]).tolist()
    models,groups=validation(table)
    # Save line layers with the new risk fields, separate from the old products.
    if (OUT/'segment_risk.gpkg').exists():
        ogr.GetDriverByName('GPKG').DeleteDataSource(str(OUT/'segment_risk.gpkg'))
    gp=ogr.GetDriverByName('GPKG').CreateDataSource(str(OUT/'segment_risk.gpkg'))
    for key in KEYS:
        ly=gp.CreateLayer(key,srs=srs,geom_type=ogr.wkbLineString)
        for name in ['segment_id','start_m','end_m','risk_score','class_code']:
            ly.CreateField(ogr.FieldDefn(name,ogr.OFTReal))
        for rec,row in zip(segments,table[table.zone.eq(key)].to_dict('records')):
            f=ogr.Feature(ly.GetLayerDefn())
            for name in ['segment_id','start_m','end_m','risk_score','class_code']: f.SetField(name,float(row[name]))
            f.SetGeometry(rec['geometry']); ly.CreateFeature(f)
    gp=None
    figures(table,groups,class_arrays,grid)
    meta=dict(zones=zone_summary,segment_count=len(segments),total_length_km=float(table[table.zone.eq('protection')].length_m.sum()/1000),
        scaling=scaling,weights=COMPONENT_WEIGHTS,environment_weights=ENVIRONMENT_WEIGHTS,insar_statistics=insar_stats,
        ratio_protection_control=zone_summary[0]['mean']/zone_summary[1]['mean'],ratio_block_ci95=ci,validation=models,
        method='Common 0-550m P5-P95 scaling; unique nearest-segment cells; score=0.7*mean+0.3*P90; fixed 20-point classes',
        missing_policy='Raw factor or InSAR-neighbourhood missing excluded; no missing-as-zero in risk results',
        limitations=['Analytical line-based zones, not surveyed or legally published boundaries',
            'Relative index, not probability or expected loss; class cutoffs inherited, not externally calibrated',
            'Two mapped landslides; original 500m landslide distance decay retained',
            '100m InSAR interpolation radius retained; not an evaluation-zone definition',
            'Validation reuses fixed coverage and 70 unique object assignments, not a new survey',
            'Circular shift compresses gaps between covered segments; exploratory one-sided spatial test',
            'Control-zone association with wall instability is not validation of hazards throughout the control zone'])
    (OUT/'experiment_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 广武长城保护范围与建设控制地带风险评价重实验','',
        '本实验替代100 m缓冲区评价口径，保留原结果不覆盖。距离由长城线位近似墙体外缘，0–50 m与50–550 m为分析区，不替代法定区划。',
        '', '## 方法', '',
        '在0–550 m共同范围内按P5–P95统一标定环境因子；环境、InSAR、已知滑坡权重分别为0.50、0.25、0.25。10 m像元指数为加权和乘100。每个像元唯一归属最近的50 m长城分段，分段风险指数=0.70×像元均值+0.30×P90；五级阈值为20、40、60、80。等级为本模型相对指数分级，不是官方风险标准。',
        '建设控制地带扣除整条长城的保护范围。附表保留相交缓冲区方案作为空间单元敏感性结果。缺失因子和无100 m邻域InSAR观测的像元不计作零风险。',
        '', '## 分区结果','']
    for key,label,z in zip(KEYS,LABELS,zone_summary):
        d=table[table.zone.eq(key)]; peak=d.loc[d.risk_score.idxmax()]
        lines.append(f'{label}：有效面积{z["area_km2"]:.4f} km²，像元平均指数{z["mean"]:.2f}；156个对应分段指数范围{d.risk_score.min():.2f}–{d.risk_score.max():.2f}。最高值位于{peak.start_m:.0f}–{peak.end_m:.0f} m。')
        for r in length_rows:
            if r['zone']==key and r['segment_count']:
                lines.append(f'- {r["class_cn"]}：{r["segment_count"]}段，对应长城长度{r["length_km"]:.3f} km（{r["length_percent"]:.1f}%）。')
        lines.append('')
    lines += [f'保护范围/建设控制地带的全域像元平均指数比={meta["ratio_protection_control"]:.3f}，8个沿程块、5000次重采样的面积加权95%区间为{ci[0]:.3f}–{ci[1]:.3f}。该区间仅反映本研究段的空间重采样变异，不覆盖权重、区划或观测误差。', '', '## 独立关联检验', '']
    for label,m in zip(LABELS,models):
        lines.append(f'{label}：{m["covered_segments"]}个有效覆盖分段、{m["objects"]}个固定失稳对象；IRR={m["irr"]:.3f}，稳健95%CI {m["irr_lower"]:.3f}–{m["irr_upper"]:.3f}；单侧循环位移p={m["circular_shift_p_one_sided"]:.3f}，两区Holm校正p={m["circular_shift_p_holm"]:.3f}。')
    lines += ['', '原100 m模型的13.52–47.13、83.3%、IRR=1.309和p=0.044不再作为新模型结果。两区长度分别描述同一条长城，不能相加。面积等级表描述区内像元，长度等级表描述该区对应的长城分段，二者不能混用。',
        '', '## 图注', '', '图1：（a）唯一近邻像元汇总得到的两区分段风险指数；（b）各区指数等数量五组与本体失稳对象密度，误差线为Poisson精确95%区间。五组不是五个风险等级。',
        '图2：两区10 m像元风险等级；EPSG:32649。距离从线位近似起算，不替代正式保护区划。', '', '## 限制', '']
    lines += ['- '+s for s in meta['limitations']]
    (OUT/'实验说明与论文结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

def audit_and_replot():
    table=pd.read_csv(OUT/'segment_risk.csv')
    groups=pd.read_csv(OUT/'validation_groups.csv')
    ds=gdal.Open(str(DEM_PATH)); grid=raster_grid(ds)
    classes={k:gdal.Open(str(OUT/f'{k}_risk_class_10m.tif')).ReadAsArray() for k in KEYS}
    figures(table,groups,classes,grid)
    checks=[]
    for key in KEYS:
        d=table[table.zone.eq(key)]
        assert len(d)==156 and d.segment_id.nunique()==156
        assert np.allclose(d.risk_score,.7*d.pixel_mean+.3*d.pixel_p90)
        assert d.pixel_count.sum()==np.count_nonzero(classes[key])
        assert np.all((d.risk_score>=0)&(d.risk_score<=100))
        rho=d.risk_score.corr(d.overlap_sensitivity_score,method='spearman')
        checks.append(dict(zone=key,passed=True,min_pixels_per_segment=int(d.pixel_count.min()),
                           max_pixels_per_segment=int(d.pixel_count.max()),
                           spearman_unique_vs_overlap=float(rho)))
    assert not np.any((classes['protection']>0)&(classes['control']>0))
    save_table('quality_checks.csv',checks)
    print(checks)

if __name__=='__main__':
    import sys
    if '--audit-replot' in sys.argv: audit_and_replot()
    else: main()
