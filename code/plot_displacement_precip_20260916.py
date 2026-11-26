"""Plot one measured slope-unit LOS displacement series with regional NASA POWER rainfall."""
from pathlib import Path
from lxml import etree
import json
import fiona
import pandas as pd
import numpy as np
from shapely.geometry import Polygon,shape
from shapely.ops import transform
from pyproj import Transformer,CRS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path('D:/project/shuozhouinflucefactor')
OUT=ROOT/'outputs/displacement_rainfall_20260916'
OUT.mkdir(parents=True,exist_ok=True)
KML=ROOT/'data/landslide/滑坡区域2.kml'
tree=etree.parse(str(KML))
coords=tree.xpath("//k:outerBoundaryIs/k:LinearRing/k:coordinates",namespaces={'k':'http://www.opengis.net/kml/2.2'})
assert coords
lonlat=[tuple(float(v) for v in item.split(',')[:2]) for item in coords[0].text.split()]
zone=Polygon(lonlat)
assert zone.is_valid and zone.area>0
gpkg=ROOT/'outputs/repeated_environment_20260915/inputs.gpkg'
with fiona.open(gpkg,layer='complete_slopes') as src:
    crs=CRS.from_user_input(src.crs_wkt or src.crs)
    tf=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    poly=transform(tf.transform,zone)
    hits=[]
    for feat in src:
        if not feat['geometry']:continue
        geom=shape(feat['geometry'])
        overlap=geom.intersection(poly).area
        if overlap>0:
            props=dict(feat['properties'])
            hits.append(dict(SU_ID=int(props['SU_ID']),overlap_m2=overlap,zone_share=overlap/poly.area,slope_share=overlap/geom.area,eligible=int(props['eligible']),rms=props.get('rms')))
hits=sorted(hits,key=lambda x:x['overlap_m2'],reverse=True)
assert hits,'Landslide zone does not intersect slope units'

rain=pd.read_csv(ROOT/'data/climate/nasa_power/climate_daily_clean_202211_202512.csv',parse_dates=['date'],encoding='utf-8-sig')
assert rain.date.is_unique and rain.precip_mm_day.notna().all()
assert (rain.precip_mm_day>=0).all()
series=pd.read_csv(ROOT/'outputs/scientific_prediction_upgrade_20260914/T40_slope_timeseries.csv',parse_dates=['date'])
for h in hits:
    sub=series[series.SU_ID.eq(h['SU_ID'])].dropna(subset=['displacement']).sort_values('date')
    h['T40_epochs']=len(sub)
    h['T40_span_days']=(sub.date.max()-sub.date.min()).days if len(sub)>1 else 0
eligible=[h for h in hits if h['eligible']==1 and h['T40_epochs']>=55 and h['zone_share']>=.1]
assert eligible,'No zone-2 slope unit has sufficient overlap and T40 epochs'
chosen=max(eligible,key=lambda x:(x['overlap_m2'],x['T40_epochs']))
ts=series[series.SU_ID.eq(chosen['SU_ID'])].dropna(subset=['displacement']).sort_values('date').copy()
ts['displacement_mm']=ts.displacement
clim=rain[(rain.date>=ts.date.min())&(rain.date<=ts.date.max())].copy()
clim['precip_30d_mm']=clim.precip_mm_day.rolling(30,min_periods=30).sum()

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
fig,(a,b)=plt.subplots(2,1,figsize=(12.5,7.2),sharex=True,gridspec_kw={'height_ratios':[1.6,1]})
a.plot(ts.date,ts.displacement_mm,color='#aa493a',linewidth=1.75,marker='o',markersize=3.7,label='T40 累计LOS位移')
a.axhline(0,color='#777',linewidth=.75)
a.set_ylabel('累计LOS位移（mm）')
a.set_title(f'与滑坡区域2相交的坡单元 SU {chosen["SU_ID"]}：位移与区域降雨时序')
a.grid(axis='y',alpha=.25)
a.legend(loc='upper left',frameon=False)
b.bar(clim.date,clim.precip_mm_day,width=1,color='#2c81aa',alpha=.68,label='逐日降雨')
b.set_ylabel('逐日降雨（mm/d）')
b.grid(axis='y',alpha=.25)
c=b.twinx()
c.plot(clim.date,clim.precip_30d_mm,color='#496d44',linewidth=1.2,label='前30日累计降雨')
c.set_ylabel('前30日累计降雨（mm）')
line1,label1=b.get_legend_handles_labels();line2,label2=c.get_legend_handles_labels()
b.legend(line1+line2,label1+label2,loc='upper left',frameon=False,ncol=2)
b.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for tick in b.get_xticklabels():tick.set_rotation(30);tick.set_ha('right')
b.set_xlabel('日期（降雨数据采用UTC日）')
fig.tight_layout(rect=(0,.09,1,.99))
fig.text(.08,.012,'降雨：NASA POWER，研究区中心点 112.794431°E、39.201233°N；网格尺度大于研究区。\n仅用于区域气象背景对照，不代表坡面实测降雨或因果验证。',fontsize=8.5,color='#444',va='bottom')
fig.savefig(OUT/'滑坡区域2_SU位移_降雨联合时序.png',dpi=260,bbox_inches='tight')
fig.savefig(OUT/'滑坡区域2_SU位移_降雨联合时序.pdf',bbox_inches='tight')
plt.close(fig)
ts[['date','SU_ID','displacement_mm']].to_csv(OUT/'T40_坡单元位移.csv',index=False,encoding='utf-8-sig')
clim[['date','precip_mm_day','precip_30d_mm']].to_csv(OUT/'NASA_POWER_降雨.csv',index=False,encoding='utf-8-sig')
audit=dict(zone='滑坡区域2',selection='maximum KML overlap among eligible complete slope units with >=55 valid T40 epochs and >=10% zone area overlap',chosen=chosen,candidates=hits,rainfall_source=json.loads((ROOT/'data/climate/nasa_power/climate_metadata.json').read_text(encoding='utf-8')),n_rain_days=len(clim),n_insar_epochs=len(ts),plot_start=str(ts.date.min().date()),plot_end=str(ts.date.max().date()),interpretation='descriptive visual alignment only; no lag optimization or causal claim')
(OUT/'绘图核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chosen':chosen,'candidates':hits,'days':len(clim)},ensure_ascii=False),flush=True)
