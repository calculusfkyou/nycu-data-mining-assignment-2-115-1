"""Fetch only the real raw inputs documented by the committed hash manifest."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    manifest=json.loads(Path('Dataset/provenance.json').read_text())
    def fetch(row):
        dest=Path('Dataset/raw/sweagent_gpt4')/(row['instance_id']+'.traj')
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256']: return
        data=urllib.request.urlopen(row['url'],timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Source drift: '+row['instance_id'])
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch,manifest['trajectories']))
    print('All 284 real raw files match their committed SHA-256 hashes.')

if __name__=='__main__': main()
