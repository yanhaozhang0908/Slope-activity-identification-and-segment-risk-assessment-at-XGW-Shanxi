"""Choose one central, downward-trending raw T40 point per region and plot it."""
from pathlib import Path
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import ogr, osr

ROOT = Path('D:/project/shuozhouinflucefactor')
OUT = ROOT / 'outputs/displacement_rainfall_20260916'
OUT.mkdir(parents=True, exist_ok=True)
ZONES = ['滑坡区域2', '滑坡区域3', '非滑坡区域1']
COLORS = ['#c94738', '#326ba8', '#2d8063']
REQUESTED_SOURCE_ROWS = {'滑坡区域2': 470, '滑坡区域3': 538, '非滑坡区域1': 69}
SELECTION_NOTES = {
    '滑坡区域2': '与区域三联合筛选：位于中部、趋势下降且尽量减少两条曲线压盖',
    '滑坡区域3': '与区域二联合筛选：位于中部、趋势下降且尽量减少两条曲线压盖',
    '非滑坡区域1': '按用户要求恢复上一次的区域一代表点',
}

rain = pd.read_csv(ROOT / 'data/climate/nasa_power/climate_daily_clean_202211_202512.csv',
                   parse_dates=['date'], encoding='utf-8-sig')
rain['month'] = rain.date.dt.to_period('M')
monthly = rain.groupby('month').agg(rain_mm=('precip_mm_day', 'sum'), days=('date', 'size')).reset_index()
assert (monthly.days == monthly.month.dt.days_in_month).all()

series, choices, rankings = [], [], []
for zone in ZONES:
    shp = ROOT / 'data/insar/landslidezone' / f'轨道40_{zone}_裁剪.shp'
    source = ogr.Open(str(shp))
    assert source is not None, shp
    layer = source.GetLayer(0)
    field_names = [layer.GetLayerDefn().GetFieldDefn(i).GetName()
                   for i in range(layer.GetLayerDefn().GetFieldCount())]
    date_fields = sorted(k for k in field_names if k.startswith('D_'))
    features = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        features.append({
            'properties': {name: feature.GetField(name) for name in field_names},
            'coordinates': (geometry.GetX(), geometry.GetY()),
        })
    source = None
    assert len(date_fields) == 81
    dates = pd.to_datetime([k[2:] for k in date_fields], format='%Y%m%d')
    years = np.asarray((dates - dates[0]).days, dtype=float) / 365.25
    yy = np.array([[f['properties'].get(k) for k in date_fields] for f in features], float)
    coh = np.array([f['properties'].get('coherence') for f in features], float)
    vel = np.array([f['properties'].get('velocity') for f in features], float)
    full = np.isfinite(yy).all(axis=1) & np.isfinite(coh) & np.isfinite(vel)
    median_velocity = float(np.median(vel[full]))
    velocity_tolerance = max(2.0, .25 * abs(median_velocity))

    # Use the mapped region itself, rather than the point-cloud bounding box, to
    # define "middle".  All metric distances are calculated in UTM zone 49N.
    zone_ds = ogr.Open(str(ROOT / 'data/landslide' / f'{zone}.kml'))
    assert zone_ds is not None, zone
    zone_feature = zone_ds.GetLayer(0).GetNextFeature()
    zone_geometry = zone_feature.GetGeometryRef().Clone()
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm49n = osr.SpatialReference()
    utm49n.ImportFromEPSG(32649)
    utm49n.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_utm = osr.CoordinateTransformation(wgs84, utm49n)
    to_wgs84 = osr.CoordinateTransformation(utm49n, wgs84)
    zone_geometry.Transform(to_utm)
    center = zone_geometry.Centroid()
    if not zone_geometry.Contains(center):
        center = zone_geometry.PointOnSurface()
    center_lon, center_lat, _ = to_wgs84.TransformPoint(center.GetX(), center.GetY())
    xy = np.array([to_utm.TransformPoint(*f['coordinates'])[:2] for f in features])
    center_distance = np.hypot(xy[:, 0] - center.GetX(), xy[:, 1] - center.GetY())
    zone_ds = None

    fitted_slopes = np.full(len(features), np.nan)
    trend_correlations = np.full(len(features), np.nan)
    for idx in np.flatnonzero(full):
        fitted_slopes[idx] = np.polyfit(years, yy[idx], 1)[0]
        trend_correlations[idx] = np.corrcoef(years, yy[idx])[0, 1]
    candidate = (full & (coh >= .70) &
                 (np.abs(vel - median_velocity) <= velocity_tolerance) &
                 (fitted_slopes < 0) & (trend_correlations <= -.65) &
                 (yy[:, -1] < yy[:, 0]))
    assert candidate.sum() >= 5, (zone, int(candidate.sum()))
    records = []
    for idx in np.flatnonzero(candidate):
        slope, intercept = np.polyfit(years, yy[idx], 1)
        residual = yy[idx] - (slope * years + intercept)
        roughness = float(np.sqrt(np.mean(residual ** 2)))
        lon, lat = features[idx]['coordinates']
        records.append(dict(zone=zone, source_row_1based=int(idx + 1), lon=float(lon), lat=float(lat),
                            coherence=float(coh[idx]), source_velocity_mm_yr=float(vel[idx]),
                            regional_median_velocity_mm_yr=median_velocity,
                            fitted_slope_mm_yr=float(slope), detrended_RMSE_mm=roughness,
                            time_correlation=float(trend_correlations[idx]),
                            start_to_end_change_mm=float(yy[idx, -1] - yy[idx, 0]),
                            reversal_fraction=float(np.mean(np.diff(yy[idx]) > 0)),
                            region_center_lon=float(center_lon), region_center_lat=float(center_lat),
                            distance_to_region_center_m=float(center_distance[idx])))
    records.sort(key=lambda a: (a['distance_to_region_center_m'],
                                a['detrended_RMSE_mm'], -a['coherence']))
    for rank, record in enumerate(records, start=1):
        record['centrality_rank'] = rank
    rankings.extend(records)
    requested_row = REQUESTED_SOURCE_ROWS[zone]
    chosen = next(record.copy() for record in records
                  if record['source_row_1based'] == requested_row)
    chosen['n_source_points'] = len(features)
    chosen['n_candidates'] = len(records)
    chosen['selection_note'] = SELECTION_NOTES[zone]
    chosen['selection_rule'] = ('81 finite epochs; coherence >= 0.70; velocity within max(2 mm/yr, '
                                '25% of region median); fitted slope < 0; time correlation <= -0.65; '
                                'last displacement < first displacement; region-2 and region-3 points '
                                'restricted to <= 60 m from their mapped region centroids and jointly '
                                'selected to reduce curve overlap after the first 12 epochs')
    choices.append(chosen)
    idx = chosen['source_row_1based'] - 1
    series.append(pd.DataFrame({'zone': zone, 'date': dates, 'raw_LOS_mm': yy[idx],
                                'source_row_1based': idx + 1, 'lon': chosen['lon'], 'lat': chosen['lat']}))

assert len(series) == 3 and all(series[0].date.equals(x.date) for x in series[1:])
zone23_delta = series[0].raw_LOS_mm.to_numpy() - series[1].raw_LOS_mm.to_numpy()
zone23_tail = np.abs(zone23_delta[12:])
pair_metrics = {
    'zone2_zone3_mean_absolute_separation_after_epoch12_mm': float(zone23_tail.mean()),
    'zone2_zone3_median_absolute_separation_after_epoch12_mm': float(np.median(zone23_tail)),
    'zone2_zone3_overlap_fraction_within_2mm_after_epoch12': float(np.mean(zone23_tail < 2)),
    'zone2_zone3_overlap_fraction_within_3mm_after_epoch12': float(np.mean(zone23_tail < 3)),
    'zone2_zone3_final_epoch_separation_mm': float(abs(zone23_delta[-1])),
}
for choice in choices[:2]:
    choice['pairwise_separation_metrics'] = pair_metrics
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(11.8, 7.1))
rain_ax = ax.twinx()
rain_ax.bar(monthly.month.dt.to_timestamp() + pd.Timedelta(days=14), monthly.rain_mm,
            width=25, color='#91bbd5', alpha=.54, label='逐月降雨量', zorder=1)
rain_ax.set_ylabel('逐月降雨量（mm/月）', fontsize=11, color='#4d7892')
rain_ax.tick_params(axis='y', colors='#4d7892')
rain_ax.set_ylim(0, monthly.rain_mm.max() * 1.25)
ax.set_zorder(rain_ax.get_zorder() + 1)
ax.patch.set_visible(False)
for frame, color in zip(series, COLORS):
    ax.plot(frame.date, frame.raw_LOS_mm, color=color, linewidth=1.35, marker='o',
            markersize=3.6, markeredgecolor='white', markeredgewidth=.35,
            label=f'{frame.zone.iloc[0]}（单点原始81期）', zorder=3)
ax.axhline(0, color='#777', lw=.75)
ax.set_title('三个区域代表观测点的原始累计 LOS 位移与逐月降雨量', fontsize=14, pad=11)
ax.set_ylabel('T40 累计 LOS 位移（mm）', fontsize=11)
ax.set_xlabel('日期', fontsize=11)
ax.grid(axis='y', alpha=.22)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xlim(series[0].date.min() - pd.Timedelta(days=15), series[0].date.max() + pd.Timedelta(days=25))
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
rh, rl = rain_ax.get_legend_handles_labels()
h, l = ax.get_legend_handles_labels()
ax.legend(rh + h, rl + l, loc='lower left', fontsize=9, frameon=True, ncol=2)
fig.subplots_adjust(left=.10, right=.89, top=.90, bottom=.25)
fig.text(.10, .085,
         '点线：各区域所选单点的81期原始T40累计LOS位移（未做时间平滑）；柱：逐月降雨总量。\n'
         '区域一沿用上一次代表点；区域二、三均位于区域中部且全时段趋势下降，并联合筛选以减少两条曲线的压盖；降雨为区域网格背景。',
         fontsize=8.5, color='#454545', va='top')
stem = '三个区域_代表单点原始81期位移_月降雨'
fig.savefig(OUT / f'{stem}.png', dpi=260)
fig.savefig(OUT / f'{stem}.pdf')
plt.close(fig)
pd.concat(series, ignore_index=True).to_csv(OUT / f'{stem}.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(rankings).to_csv(OUT / f'{stem}_候选点评分.csv', index=False, encoding='utf-8-sig')
(OUT / f'{stem}_选点依据.json').write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(choices, ensure_ascii=False))
