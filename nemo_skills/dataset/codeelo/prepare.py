import json
import argparse
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Path for the output JSONL file.")
    args = parser.parse_args()

    dataset = load_dataset("Qwen/CodeElo", split="test")

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for x, entry in enumerate(dataset):
            problem_id = entry['problem_id']

            example_blocks = []
            for example_pair in entry['examples']:
                input_text, output_text = example_pair
                example_blocks.append(f"Input\n\n{input_text.strip()}\n\nOutput\n\n{output_text.strip()}")

            examples_str = "Examples\n\n" + "\n\n".join(example_blocks)

            question = (
                f"{entry['description'].strip()}\n\n"
                f"Input\n\n{entry['input'].strip()}\n\n"
                f"Output\n\n{entry['output'].strip()}\n\n"
                f"{examples_str}\n\n"
                f"Note\n\n{entry['note'].strip()}"
            )

            output_entry = {
                "id": x,
                "cf_contest_id": int(problem_id[:4]),
                "cf_index": problem_id[4:],
                "question": question
            }

            f_out.write(json.dumps(output_entry) + '\n')
