import os, time, re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

BASE = "/Users/dingyizhuang/MIT Dropbox/Dingyi Zhuang/geo-llm-bias"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

oai = OpenAI(api_key=OPENAI_API_KEY)

queries = pd.read_csv(f"{BASE}/data/processed/query_set.csv")
out_path = f"{BASE}/data/processed/llm_responses_gpt.csv"

def extract_number(text):
    if not text: return None
    matches = re.findall(r'\d+\.?\d*', str(text).replace(',',''))
    return float(matches[0]) if matches else None

# Resume if partial results exist
if os.path.exists(out_path):
    existing = pd.read_csv(out_path)
    done_ids = set(zip(existing['geonameid'].astype(str), existing['query_type']))
    remaining = queries[~queries.apply(lambda r: (str(r['geonameid']), r['query_type']) in done_ids, axis=1)]
    results = existing.to_dict('records')
    print(f"Resuming: {len(existing)} done, {len(remaining)} remaining")
else:
    remaining = queries
    results = []
    print(f"Starting fresh: {len(remaining)} queries")

for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="GPT-4o-mini"):
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": row['prompt']}],
            temperature=0,
            max_tokens=15
        )
        answer = resp.choices[0].message.content.strip()
        results.append({**row.to_dict(), 'model': 'gpt-4o-mini', 'raw_response': answer, 'numeric_response': extract_number(answer)})
    except Exception as e:
        print(f"Error on {row['city']}: {e}")
        results.append({**row.to_dict(), 'model': 'gpt-4o-mini', 'raw_response': f"ERROR: {e}", 'numeric_response': None})
    
    time.sleep(0.3)
    
    if len(results) % 50 == 0:
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"  Saved {len(results)} results so far...")

pd.DataFrame(results).to_csv(out_path, index=False)
print(f"Done! Total: {len(results)} responses")
valid = sum(1 for r in results if r['numeric_response'] is not None)
print(f"Valid numeric responses: {valid}/{len(results)}")
