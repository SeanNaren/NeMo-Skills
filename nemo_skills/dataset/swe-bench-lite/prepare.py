import argparse
import json
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset to a JSONL file.")
parser.add_argument('--dataset_name', type=str, default='princeton-nlp/SWE-bench_Lite')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--output_file', type=str, default='test.jsonl')
args = parser.parse_args()

dataset = load_dataset(args.dataset_name, split=args.split)

with open(args.output_file, 'w', encoding='utf-8') as f:
    for record in dataset:
        f.write(json.dumps(record) + '\n')

print(f"Dataset '{args.dataset_name}' (split: {args.split}) converted to '{args.output_file}'.")
