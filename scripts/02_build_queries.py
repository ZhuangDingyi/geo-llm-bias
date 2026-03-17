import pandas as pd, pycountry, os

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"
cities = pd.read_csv(f"{BASE}/data/raw/cities_sample.csv")

def get_country_name(cc):
    try:
        c = pycountry.countries.get(alpha_2=cc)
        return c.name if c else cc
    except:
        return cc

cities['country_name'] = cities['country_code'].apply(get_country_name)

queries = []
for _, row in cities.iterrows():
    city, country, gid = row['name'], row['country_name'], row['geonameid']
    for qtype, lang, prompt in [
        ('qol_en', 'en', f"On a scale of 1 to 10, rate the overall quality of life in {city}, {country}. Respond with only a single number between 1 and 10."),
        ('qol_zh', 'zh', f"请用1到10分评价{city}，{country}的整体生活质量。只回答一个1到10之间的数字。"),
        ('pop_en', 'en', f"What is the approximate population of {city}, {country}? Respond with only a number in millions (e.g., 2.5). No units, no text."),
    ]:
        queries.append({'geonameid': gid, 'city': city, 'country': country, 'query_type': qtype, 'language': lang, 'prompt': prompt})

pd.DataFrame(queries).to_csv(f"{BASE}/data/processed/query_set.csv", index=False)
print(f"Query set: {len(queries)} prompts for {len(cities)} cities")
print(f"Sample query:\n{queries[0]['prompt']}")
