#!/usr/bin/env python3
"""Append compact progress heartbeats for the MLP tuning pipeline."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    root, log = Path(args.root), Path(args.log)
    while True:
        parts, evaluation_complete = [], 0
        for dataset in args.datasets:
            summary = read_json(root / dataset / 'tuning' / 'summary.json')
            if summary:
                parts.append('{}:{}/{}'.format(
                    dataset, summary.get('successful_cells', 0),
                    summary.get('required_cells', 0)))
            else:
                parts.append('{}:waiting'.format(dataset))
            evaluation = root / dataset / 'evaluation_results.jsonl'
            if evaluation.exists():
                shyre_seeds = set()
                for line in evaluation.read_text(encoding='utf-8').splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get('status') == 'success' and row.get('method') == 'SHyRe':
                        shyre_seeds.add(row.get('seed'))
                if len(shyre_seeds) >= 3:
                    evaluation_complete += 1
        line = '{} HEARTBEAT tuning=[{}] evaluated={}/{}\n'.format(
            datetime.now().astimezone().isoformat(), ', '.join(parts),
            evaluation_complete, len(args.datasets))
        with log.open('a', encoding='utf-8') as handle:
            handle.write(line)
        print(line, end='', flush=True)
        status = read_json(root / 'status.json')
        if status.get('complete') or evaluation_complete == len(args.datasets):
            break
        time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()
