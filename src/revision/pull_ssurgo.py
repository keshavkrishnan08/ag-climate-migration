"""R2-1b: pull county-level SSURGO soil properties via USDA Soil Data Access REST."""
import json, requests, pandas as pd, numpy as np
from pathlib import Path
OUT = Path("results/revision"); OUT.mkdir(parents=True, exist_ok=True)
# area-weighted available water storage (0-150cm) and clay% by survey area (areasymbol -> county FIPS)
q = ("SELECT l.areasymbol, "
     "SUM(mu.muacres) AS acres, "
     "SUM(m.aws0150wta*mu.muacres)/NULLIF(SUM(mu.muacres),0) AS aws0150, "
     "SUM(m.claytotal_r*mu.muacres)/NULLIF(SUM(mu.muacres),0) AS clay "
     "FROM legend l JOIN mapunit mu ON mu.lkey=l.lkey "
     "JOIN muaggatt m ON m.mukey=mu.mukey "
     "GROUP BY l.areasymbol")
url = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
try:
    r = requests.post(url, json={"format": "JSON+COLUMNNAME", "query": q}, timeout=120)
    print("status", r.status_code)
    data = r.json().get("Table", [])
    if not data:
        # muaggatt may lack claytotal_r; retry simpler
        q2 = ("SELECT l.areasymbol, SUM(mu.muacres) acres, "
              "SUM(m.aws0150wta*mu.muacres)/NULLIF(SUM(mu.muacres),0) aws0150 "
              "FROM legend l JOIN mapunit mu ON mu.lkey=l.lkey JOIN muaggatt m ON m.mukey=mu.mukey "
              "GROUP BY l.areasymbol")
        r = requests.post(url, json={"format": "JSON+COLUMNNAME", "query": q2}, timeout=120)
        print("retry status", r.status_code); data = r.json().get("Table", [])
    cols = data[0]; rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)
    # areasymbol like 'IA021' -> state alpha + county code; map to FIPS via state alpha
    st = {'AL':'01','AK':'02','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
    df = df[df['areasymbol'].str[:2].isin(st)].copy()
    df['fips'] = df['areasymbol'].str[:2].map(st) + df['areasymbol'].str[2:5]
    for c in ['acres','aws0150'] + (['clay'] if 'clay' in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    cty = df.groupby('fips').apply(lambda d: pd.Series({
        'aws0150': np.average(d['aws0150'].dropna(), weights=d.loc[d['aws0150'].notna(),'acres']) if d['aws0150'].notna().any() and d['acres'].sum()>0 else np.nan,
        'clay': (np.average(d['clay'].dropna(), weights=d.loc[d['clay'].notna(),'acres']) if 'clay' in d and d['clay'].notna().any() else np.nan)
    }), include_groups=False).reset_index()
    cty.to_parquet(OUT/'ssurgo_county_soil.parquet', index=False)
    print("SSURGO counties:", len(cty), "| sample:", cty.head(3).to_dict('records'))
except Exception as e:
    import traceback; traceback.print_exc(); print("SSURGO pull failed:", e)
