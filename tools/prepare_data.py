"""Re-extract cached real trajectories; preserve the previously inspected test IDs.

Run tools/fetch_raw.py first when the ignored raw cache is absent.
Never manufacture missing trajectories or labels.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Dataset.download_and_extract import extract_features_from_trajectory, S3_BASE_URL, RESULTS_JSON_URL


def main():
    dest = ROOT / 'Dataset'
    prior = pd.read_csv(dest / 'sweagent_gpt4_features.csv')
    split_path = dest / 'split_ids.json'
    if split_path.exists():
        splits = json.loads(split_path.read_text())
    else:
        # Keep the existing test membership; move 50 development failures to train.
        train = pd.read_csv(dest / 'train.csv')
        val = pd.read_csv(dest / 'val.csv')
        test = pd.read_csv(dest / 'test.csv')
        moved = val[val.label == 1].sample(n=50, random_state=42)
        splits = {'train': train.instance_id.tolist() + moved.instance_id.tolist(),
                  'val': val[~val.instance_id.isin(moved.instance_id)].instance_id.tolist(),
                  'test': test.instance_id.tolist()}
        split_path.write_text(json.dumps(splits, indent=2) + '\n')
    records, provenance = [], []
    for row in prior.itertuples():
        raw = dest / 'raw/sweagent_gpt4' / f'{row.instance_id}.traj'
        content = raw.read_bytes()
        data = json.loads(content)
        rec = extract_features_from_trajectory(data, row.instance_id, bool(row.is_resolved))
        records.append(rec)
        provenance.append({'instance_id': row.instance_id, 'sha256': hashlib.sha256(content).hexdigest(),
                           'url': f'{S3_BASE_URL}/{row.instance_id}.traj'})
    df = pd.DataFrame(records)
    df.to_csv(dest / 'sweagent_gpt4_features.csv', index=False)
    for name, ids in splits.items():
        df.set_index('instance_id').loc[ids].reset_index().to_csv(dest / f'{name}.csv', index=False)
    (dest / 'provenance.json').write_text(json.dumps({'label_source': RESULTS_JSON_URL,
        'label_note': 'Labels cross-checked against archived official results.json by verify_submission.py.',
        'trajectories': provenance}, indent=2) + '\n')
    train = df[df.instance_id.isin(splits['train'])]
    train.groupby('label')[list(df.columns[5:])].agg(['mean', 'std']).to_csv(dest / 'summary_statistics.csv')
    print({k: len(v) for k, v in splits.items()})


if __name__ == '__main__':
    main()
