"""Plot 81 T40 LOS epochs against rainfall totals of their acquisition months."""
from pathlib import Path
import json

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform

ROOT = Path('D:/project/shuozhouinflucefactor')
OUT = ROOT / 'outputs/displacement_rainfall_20260916'
OUT.mkdir(parents=True, exist_ok=True)
ZONES = ['滑坡区域2', '滑坡区域3', '非滑坡区域1']
COLORS = ['#c94738', '#326ba8', '#2d8063']

rain = pd.read_csv(ROOT / 'data/climate/nasa_power/climate_daily_clean_202211_202512.csv',
                   parse_dates=['date'], encoding='utf-8-sig')
assert rain.date.is_unique and rain.precip_mm_day.notna().all() and rain.precip_mm_day.ge(0).all()
rain['month'] = rain.date.dt.to_period('M')
monthly = rain.groupby('month').agg(monthly_rain_mm=('precip_mm_day', 'sum'),
                                    days=('date', 'size')).reset_index()
assert (monthly.days == monthly.month.dt.days_in_month).all(), 'Monthly rain total has incomplete months'

ts = pd.read_csv(ROOT / 'outputs/scientific_prediction_upgrade_20260914/T40_slope_timeseries.csv',
                 parse_dates=['date'])
ts = ts.dropna(subset=['displacement']).sort_values('date')
counts = ts.groupby('SU_ID').date.nunique()
all_dates = ts.date.drop_duplicates().sort_values()
assert len(all_dates) == 81

gpkg = ROOT / 'outputs/repeated_environment_20260915/inputs.gpkg'
with fiona.open(gpkg, layer='complete_slopes') as src:
    crs = CRS.from_user_input(src.crs_wkt or src.crs)
    slopes = [(int(feat['properties']['SU_ID']), int(feat['properties']['eligible']),
               shape(feat['geometry'])) for feat in src if feat['geometry']]
project = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
selected, rows = [], []
for zone_name in ZONES:
    tree = etree.parse(str(ROOT / 'data/landslide' / f'{zone_name}.kml'))
    coords = tree.xpath('//k:outerBoundaryIs/k:LinearRing/k:coordinates',
                        namespaces={'k': 'http://www.opengis.net/kml/2.2'})
    assert coords, zone_name
    lonlat = [tuple(float(v) for v in item.split(',')[:2]) for item in coords[0].text.split()]
    polygon = transform(project.transform, Polygon(lonlat))
    assert polygon.is_valid and polygon.area > 0
    candidates = []
    for suid, eligible, geom in slopes:
        area = geom.intersection(polygon).area
        if area > 0 and eligible == 1 and counts.get(suid, 0) == 81:
            candidates.append((area, suid, area / polygon.area, area / geom.area))
    assert candidates, f'No 81-epoch eligible slope overlaps {zone_name}'
    area, suid, zone_share, slope_share = max(candidates)
    d = ts.loc[ts.SU_ID.eq(suid), ['date', 'displacement']].copy()
    assert len(d) == 81 and set(d.date) == set(all_dates), f'Incomplete date series: {zone_name}'
    d['month'] = d.date.dt.to_period('M')
    d = d.merge(monthly[['month', 'monthly_rain_mm']], on='month', how='left', validate='many_to_one')
    assert d.monthly_rain_mm.notna().all()
    d.insert(0, 'zone', zone_name)
    d.insert(1, 'SU_ID', suid)
    selected.append(dict(zone=zone_name, SU_ID=suid, overlap_m2=area,
                         zone_share=zone_share, slope_share=slope_share, n_epochs=len(d)))
    rows.append(d)

data = pd.concat(rows, ignore_index=True)
assert len(data) == 243
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10.8, 6.7))
for zone_name, color in zip(ZONES, COLORS):
    d = data[data.zone.eq(zone_name)]
    suid = int(d.SU_ID.iloc[0])
    ax.scatter(d.monthly_rain_mm, d.displacement, s=34, color=color, alpha=.72,
               edgecolors='white', linewidths=.35, label=f'{zone_name}（SU {suid}，81期）')
ax.axhline(0, color='#777', lw=.8)
ax.set_xlabel('影像所在月份的降雨总量（mm/月）', fontsize=11)
ax.set_ylabel('T40 累计 LOS 位移（mm）', fontsize=11)
ax.set_title('三个区域的累计 LOS 位移与逐月降雨量', fontsize=14, pad=12)
ax.grid(alpha=.23)
ax.legend(loc='best', frameon=True, fontsize=9.3)
fig.subplots_adjust(left=.11, right=.98, top=.90, bottom=.23)
fig.text(.11, .085,
         '每点为一期影像；同月多期影像共用该月降雨总量。位移为所选相交坡单元的平均累计 LOS 位移。\n'
         '降雨为 NASA POWER 研究区中心点网格资料，仅表示区域背景；本图不用于判定降雨致因。',
         fontsize=8.8, color='#484848', va='top')
stem = '三个区域_逐月降雨量_81期累计LOS位移'
fig.savefig(OUT / f'{stem}.png', dpi=260)
fig.savefig(OUT / f'{stem}.pdf')
plt.close(fig)
data[['zone', 'SU_ID', 'date', 'monthly_rain_mm', 'displacement']].to_csv(
    OUT / f'{stem}.csv', index=False, encoding='utf-8-sig')
(OUT / f'{stem}_核查.json').write_text(
    json.dumps({'selection': selected, 'monthly_rain_source': 'NASA POWER PRECTOTCORR UTC daily sum',
                'months': len(monthly), 'n_points': len(data),
                'date_start': str(all_dates.min().date()), 'date_end': str(all_dates.max().date()),
                'interpretation': 'Descriptive scatter only; cumulative displacement and same-month rain do not establish rainfall causality.'},
               ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(selected, ensure_ascii=False))
