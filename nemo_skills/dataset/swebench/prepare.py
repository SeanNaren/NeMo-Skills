import argparse
import json
from datasets import load_dataset


def convert_hf_to_jsonl(dataset_name, split, output_file):
    dataset = load_dataset(dataset_name, split=split)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in dataset:
            json_record = json.dumps(record)
            f.write(json_record + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Verified"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test.jsonl"
    )
    args = parser.parse_args()
    convert_hf_to_jsonl(args.dataset_name, args.split, args.output_file)
