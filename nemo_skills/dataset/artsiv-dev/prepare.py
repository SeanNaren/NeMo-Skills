import argparse
import json
import random
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def main():
    parser = argparse.ArgumentParser(description="Create a dev dataset by sampling from a train.jsonl file.")
    parser.add_argument('--input_file', type=str, default='train.jsonl', help='Input JSONL file to sample from')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to take from the dataset')
    parser.add_argument('--output_file', type=str, default='test.jsonl')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Loading dataset from: {args.input_file}")
    
    # Read JSONL file
    dataset_samples = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    dataset_samples.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return
    
    print(f"Dataset size: {len(dataset_samples)}")
    
    # Randomly sample from the dataset
    if len(dataset_samples) < args.num_samples:
        print(f"Warning: Dataset has only {len(dataset_samples)} samples, using all of them")
        sampled_data = dataset_samples
    else:
        sampled_indices = random.sample(range(len(dataset_samples)), args.num_samples)
        sampled_data = [dataset_samples[i] for i in sampled_indices]
    
    # Add source information to each sample
    for sample in sampled_data:
        sample['source_dataset'] = 'input_file'
    
    # Use the samples directly
    combined_samples = sampled_data
    
    # Ensure uniqueness based on a unique identifier
    # We'll use instance_id if available, otherwise use a combination of fields
    seen_ids = set()
    unique_samples = []
    
    for sample in combined_samples:
        # Try to find a unique identifier
        unique_id = None
        if 'instance_id' in sample:
            unique_id = sample['instance_id']
        elif 'id' in sample:
            unique_id = sample['id']
        else:
            # Create a unique identifier from available fields
            # This is a fallback - we'll use a hash of some key fields
            key_fields = []
            if 'problem_statement' in sample:
                key_fields.append(str(sample['problem_statement'])[:100])  # First 100 chars
            if 'patch' in sample:
                key_fields.append(str(sample['patch'])[:100])  # First 100 chars
            if 'repo' in sample:
                key_fields.append(str(sample['repo']))
            unique_id = hash('|'.join(key_fields))
        
        if unique_id not in seen_ids:
            seen_ids.add(unique_id)
            unique_samples.append(sample)
        else:
            print(f"Duplicate found and removed: {unique_id}")
    
    # Shuffle the combined samples for good measure
    random.shuffle(unique_samples)
    
    # Write to JSONL file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for record in unique_samples:
            f.write(json.dumps(record, cls=DateTimeEncoder) + '\n')
    
    print(f"Created dev dataset with {len(unique_samples)} unique samples:")
    print(f"  - Input samples: {len(sampled_data)}")
    print(f"  - Total unique samples: {len(unique_samples)}")
    print(f"  - Saved to: {args.output_file}")
    print(f"  - Random seed used: {args.seed}")

if __name__ == "__main__":
    main()
