from pathlib import Path
import json
from osgeo import ogr
ROOT=Path('D:/project/shuozhouinflucefactor')
out=ROOT/'outputs/scientific_prediction_upgrade_20260914';out.mkdir(parents=True,exist_ok=True)
rows=[]
for p in (ROOT/'data/insar').rglob('*.shp'):
    ds=ogr.Open(str(p));l=ds.GetLayer();de=l.GetLayerDefn();fields=[de.GetFieldDefn(i).GetName() for i in range(de.GetFieldCount())]
    row=dict(path=str(p),n=l.GetFeatureCount(),fields=fields);rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
(out/'insar_schema_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
