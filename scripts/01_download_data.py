import os, requests, zipfile, io, time
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"
os.makedirs(f"{BASE}/data/raw", exist_ok=True)
os.makedirs(f"{BASE}/data/processed", exist_ok=True)
os.makedirs(f"{BASE}/figures", exist_ok=True)
os.makedirs(f"{BASE}/paper", exist_ok=True)
os.makedirs(f"{BASE}/scripts", exist_ok=True)

# --- GeoNames cities15000 ---
cities_path = f"{BASE}/data/raw/cities15000.txt"
if not os.path.exists(cities_path):
    print("Downloading GeoNames...")
    r = requests.get("http://download.geonames.org/export/dump/cities15000.zip", timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extract("cities15000.txt", f"{BASE}/data/raw/")
else:
    print("GeoNames already exists, skipping download.")

cols = ['geonameid','name','asciiname','alternatenames','latitude','longitude',
        'feature_class','feature_code','country_code','cc2','admin1_code',
        'admin2_code','admin3_code','admin4_code','population','elevation',
        'dem','timezone','modification_date']
cities = pd.read_csv(cities_path, sep='\t', names=cols, low_memory=False)
cities = cities[cities['population'] > 200000].copy()
print(f"Cities with pop>200k: {len(cities)}")

continent_map = {
    'NG':'Africa','ET':'Africa','EG':'Africa','CD':'Africa','TZ':'Africa','KE':'Africa',
    'ZA':'Africa','UG':'Africa','GH':'Africa','MZ':'Africa','MG':'Africa','CI':'Africa',
    'CM':'Africa','AO':'Africa','NE':'Africa','ML':'Africa','ZM':'Africa','MW':'Africa',
    'SN':'Africa','ZW':'Africa','SO':'Africa','TN':'Africa','DZ':'Africa','MA':'Africa',
    'LY':'Africa','SD':'Africa','RW':'Africa','BJ':'Africa','TD':'Africa',
    'US':'North America','CA':'North America','MX':'North America','GT':'North America',
    'CU':'North America','DO':'North America','HN':'North America','SV':'North America',
    'NI':'North America','CR':'North America','PA':'North America','HT':'North America',
    'JM':'North America',
    'BR':'South America','CO':'South America','AR':'South America','PE':'South America',
    'VE':'South America','CL':'South America','EC':'South America','BO':'South America',
    'PY':'South America','UY':'South America','GY':'South America','SR':'South America',
    'CN':'Asia','IN':'Asia','ID':'Asia','PK':'Asia','BD':'Asia','JP':'Asia',
    'PH':'Asia','VN':'Asia','TR':'Asia','IR':'Asia','TH':'Asia','MM':'Asia',
    'KR':'Asia','IQ':'Asia','AF':'Asia','SA':'Asia','UZ':'Asia','MY':'Asia',
    'YE':'Asia','NP':'Asia','KZ':'Asia','SY':'Asia','KH':'Asia','AE':'Asia',
    'TW':'Asia','HK':'Asia','SG':'Asia','KP':'Asia','LK':'Asia','IL':'Asia',
    'JO':'Asia','AZ':'Asia','TJ':'Asia','GE':'Asia','LB':'Asia','KW':'Asia',
    'OM':'Asia','QA':'Asia','BH':'Asia',
    'RU':'Europe','DE':'Europe','GB':'Europe','FR':'Europe','IT':'Europe',
    'ES':'Europe','UA':'Europe','PL':'Europe','RO':'Europe','NL':'Europe',
    'BE':'Europe','CZ':'Europe','GR':'Europe','PT':'Europe','SE':'Europe',
    'HU':'Europe','AT':'Europe','CH':'Europe','BG':'Europe','DK':'Europe',
    'FI':'Europe','SK':'Europe','NO':'Europe','HR':'Europe','RS':'Europe',
    'BA':'Europe','AL':'Europe','MK':'Europe','SI':'Europe','MD':'Europe',
    'LT':'Europe','LV':'Europe','EE':'Europe','BY':'Europe',
    'AU':'Oceania','NZ':'Oceania','PG':'Oceania','FJ':'Oceania',
}
cities['continent'] = cities['country_code'].map(continent_map).fillna('Other')

# Stratified sample by continent
target = 500
continent_counts = cities['continent'].value_counts()
sample_sizes = (continent_counts / continent_counts.sum() * target).round().astype(int)

sampled = []
for continent, n in sample_sizes.items():
    subset = cities[cities['continent'] == continent]
    n_actual = min(n, len(subset))
    sampled.append(subset.nlargest(n_actual, 'population'))

sample_df = pd.concat(sampled).drop_duplicates('geonameid').reset_index(drop=True)
print(f"Sample size: {len(sample_df)}")
print(sample_df['continent'].value_counts())
sample_df.to_csv(f"{BASE}/data/raw/cities_sample.csv", index=False)
print(f"Saved cities_sample.csv")

# --- WorldBank GDP per capita ---
wb_gdp_path = f"{BASE}/data/raw/worldbank_gdp.csv"
if not os.path.exists(wb_gdp_path):
    print("Fetching WorldBank GDP...")
    countries = sample_df['country_code'].unique()
    gdp_data = []
    for cc in tqdm(countries):
        try:
            url = f"https://api.worldbank.org/v2/country/{cc}/indicator/NY.GDP.PCAP.CD?format=json&mrv=3"
            r = requests.get(url, timeout=10).json()
            if len(r) > 1 and r[1]:
                for entry in r[1]:
                    if entry.get('value'):
                        gdp_data.append({'country_code': cc, 'gdp_per_capita': entry['value']})
                        break
        except Exception as e:
            print(f"  GDP error for {cc}: {e}")
        time.sleep(0.1)
    pd.DataFrame(gdp_data).to_csv(wb_gdp_path, index=False)
    print(f"GDP data: {len(gdp_data)} countries")
else:
    print("GDP data already exists, skipping.")

# --- WorldBank Internet penetration ---
wb_net_path = f"{BASE}/data/raw/worldbank_internet.csv"
if not os.path.exists(wb_net_path):
    print("Fetching WorldBank Internet penetration...")
    countries = sample_df['country_code'].unique()
    net_data = []
    for cc in tqdm(countries):
        try:
            url = f"https://api.worldbank.org/v2/country/{cc}/indicator/IT.NET.USER.ZS?format=json&mrv=3"
            r = requests.get(url, timeout=10).json()
            if len(r) > 1 and r[1]:
                for entry in r[1]:
                    if entry.get('value'):
                        net_data.append({'country_code': cc, 'internet_pct': entry['value']})
                        break
        except Exception as e:
            print(f"  Internet error for {cc}: {e}")
        time.sleep(0.1)
    pd.DataFrame(net_data).to_csv(wb_net_path, index=False)
    print(f"Internet data: {len(net_data)} countries")
else:
    print("Internet data already exists, skipping.")

# --- Wikipedia pageviews ---
wiki_path = f"{BASE}/data/raw/wikipedia_pageviews.csv"
if not os.path.exists(wiki_path):
    print("Fetching Wikipedia pageviews (this takes ~10 min)...")
    wiki_data = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        city_name = row['name'].replace(' ', '_')
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{city_name}/monthly/2024010100/2024123100"
            r = requests.get(url, timeout=10, headers={'User-Agent': 'GeoLLMBias-Research/1.0 (academic)'}).json()
            total_views = sum(item['views'] for item in r.get('items', []))
        except:
            total_views = 0
        wiki_data.append({'geonameid': row['geonameid'], 'name': row['name'], 'wiki_pageviews': total_views})
        time.sleep(0.1)
    pd.DataFrame(wiki_data).to_csv(wiki_path, index=False)
    print(f"Wikipedia data: {len(wiki_data)} cities")
else:
    print("Wikipedia data already exists, skipping.")

print("\nStep 1 complete!")
print(f"Files in data/raw: {os.listdir(f'{BASE}/data/raw')}")
