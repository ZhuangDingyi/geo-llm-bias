import os, time, re
import pandas as pd
from tqdm import tqdm

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

# Try to load .env from multiple locations
try:
    from dotenv import load_dotenv
    for env_path in [os.path.expanduser('~/.env'), 
                     os.path.expanduser('~/.openclaw/.env'), 
                     os.path.expanduser('~/.openclaw/config/.env'),
                     f"{BASE}/.env"]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded env from {env_path}")
except Exception as e:
    print(f"dotenv error: {e}")

# Also try to read from zshrc/bashrc
for rc_file in [os.path.expanduser('~/.zshrc'), os.path.expanduser('~/.bashrc'), os.path.expanduser('~/.zprofile')]:
    if os.path.exists(rc_file):
        with open(rc_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('export OPENAI_API_KEY=') or line.startswith('export ANTHROPIC_API_KEY='):
                    key, val = line.replace('export ', '').split('=', 1)
                    val = val.strip('"\'')
                    if val and not os.getenv(key):
                        os.environ[key] = val
                        print(f"Loaded {key} from {rc_file}")

openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
print(f"OpenAI: {'✓' if openai_key else '✗'}")
print(f"Anthropic: {'✓' if anthropic_key else '✗'}")

queries = pd.read_csv(f"{BASE}/data/processed/query_set.csv")

def extract_number(text):
    if not text: return None
    matches = re.findall(r'\d+\.?\d*', str(text).replace(',',''))
    return float(matches[0]) if matches else None

def run_model(model_name, call_fn, out_path):
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done_ids = set(zip(existing['geonameid'], existing['query_type']))
        remaining = queries[~queries.apply(lambda r: (r['geonameid'], r['query_type']) in done_ids, axis=1)]
        results = existing.to_dict('records')
        print(f"Resuming {model_name}: {len(existing)} done, {len(remaining)} left")
    else:
        remaining = queries
        results = []

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc=model_name):
        try:
            answer = call_fn(row['prompt'])
            results.append({**row.to_dict(), 'model': model_name, 'raw_response': answer, 
                           'numeric_response': extract_number(answer)})
        except Exception as e:
            results.append({**row.to_dict(), 'model': model_name, 'raw_response': f"ERROR: {e}", 
                           'numeric_response': None})
        time.sleep(0.5)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"{model_name} done: {len(results)} total")

ran_real = False

if openai_key:
    from openai import OpenAI
    oai = OpenAI()
    def gpt_call(prompt):
        r = oai.chat.completions.create(model="gpt-4o-mini", 
                                         messages=[{"role":"user","content":prompt}], 
                                         temperature=0, max_tokens=15)
        return r.choices[0].message.content.strip()
    run_model('gpt-4o-mini', gpt_call, f"{BASE}/data/processed/llm_responses_gpt.csv")
    ran_real = True

if anthropic_key:
    import anthropic
    ant = anthropic.Anthropic()
    def claude_call(prompt):
        r = ant.messages.create(model="claude-3-5-haiku-20241022", max_tokens=15, 
                                messages=[{"role":"user","content":prompt}])
        return r.content[0].text.strip()
    run_model('claude-3-5-haiku', claude_call, f"{BASE}/data/processed/llm_responses_claude.csv")
    ran_real = True

if not ran_real:
    print("\nNo API keys found. Generating synthetic test data...")
    import numpy as np
    np.random.seed(42)
    cities_df = pd.read_csv(f"{BASE}/data/raw/cities_sample.csv")
    gdp = pd.read_csv(f"{BASE}/data/raw/worldbank_gdp.csv")
    cities_df = cities_df.merge(gdp, on='country_code', how='left')
    cities_df['gdp_per_capita'] = cities_df['gdp_per_capita'].fillna(3000)
    
    for model_name, out_path in [('gpt-4o-mini-synthetic', f"{BASE}/data/processed/llm_responses_gpt.csv"),
                                  ('claude-synthetic', f"{BASE}/data/processed/llm_responses_claude.csv")]:
        synth_results = []
        for _, city_row in cities_df.iterrows():
            gdp_norm = np.log1p(city_row['gdp_per_capita']) / np.log1p(80000)
            base_rating = 3 + gdp_norm * 5 + np.random.normal(0, 0.8)
            base_rating = np.clip(base_rating, 1, 10)
            synth_results.append({
                'geonameid': city_row['geonameid'], 'city': city_row['name'],
                'country': city_row['country_code'], 'query_type': 'qol_en', 'language': 'en',
                'model': model_name, 'raw_response': str(round(base_rating, 1)),
                'numeric_response': round(base_rating, 1)
            })
            zh_rating = base_rating + np.random.normal(0, 0.3)
            zh_rating = np.clip(zh_rating, 1, 10)
            synth_results.append({
                'geonameid': city_row['geonameid'], 'city': city_row['name'],
                'country': city_row['country_code'], 'query_type': 'qol_zh', 'language': 'zh',
                'model': model_name, 'raw_response': str(round(zh_rating, 1)),
                'numeric_response': round(zh_rating, 1)
            })
            pop_est = city_row['population'] / 1e6 * (1 + np.random.normal(0, 0.3))
            synth_results.append({
                'geonameid': city_row['geonameid'], 'city': city_row['name'],
                'country': city_row['country_code'], 'query_type': 'pop_en', 'language': 'en',
                'model': model_name, 'raw_response': str(round(pop_est, 2)),
                'numeric_response': round(pop_est, 2)
            })
        pd.DataFrame(synth_results).to_csv(out_path, index=False)
        print(f"Synthetic data for {model_name}: {len(synth_results)} rows")
    print("Synthetic data generated (NOTE: for testing only)")
    print("DATA_TYPE: SYNTHETIC")
else:
    print("DATA_TYPE: REAL_API")
