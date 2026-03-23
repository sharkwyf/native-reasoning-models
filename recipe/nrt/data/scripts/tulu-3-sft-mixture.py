#
# Copyright 2026 Yuanfu Wang
# Modified by Yuanfu Wang (Shanghai Artificial Intelligence)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Preprocess the Tulu-3 SFT Mixture dataset to parquet format
"""

import argparse
import os
import re
from typing import List, Dict, Any

import datasets
from template import construct_messages, construct_reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_ver', type=str, default='v1', help='Prompt version to use')
    parser.add_argument("--input_dir", default="./recipe/nrt/data/raw/tulu-3-sft-mixture/")
    parser.add_argument('--max_response_length', type=int, default=-1, help='Max response length')
    parser.add_argument("--train_size", type=int, default=220000, help='Number of examples to use for training')
    parser.add_argument('--test_size', type=int, default=1024, help='Number of examples to use for testing')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of examples to sample')
    parser.add_argument('--num_proc', type=int, default=100, help='Number of processes to use for processing')
    parser.add_argument("--output_dir", default="./recipe/nrt/data/processed/tulu-3-sft-mixture-processed")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()

    # Load all parquet files from the directory
    file_pattern = os.path.join(args.input_dir, "data/*.parquet")
    raw_dataset = datasets.load_dataset('parquet', data_files=file_pattern, split="train")

    # First split to get test set
    split_dataset = raw_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
    
    # Limit training dataset to train_size if specified
    if args.train_size > 0 and len(train_dataset) > args.train_size:
        train_dataset = train_dataset.select(range(args.train_size))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    def process_fn(example, idx):
        """Process each conversation in the dataset"""
        
        messages = example.get("messages", [])
        if messages[-2]["role"] != "user":
            print(f"Warning: Last user message is not the last message, got {messages[-2]['role']}")
            return None
        if messages[-1]["role"] != "assistant":
            print(f"Warning: Last message is not from assistant, got {messages[-1]['role']}")
            return None

        prompt_messages = messages[:-2]  # All messages except the last assistant message
        question = messages[-2]["content"]
        ground_truth = messages[-1]["content"]

        constructed_messages = construct_messages(args.prompt_ver, previous_messages=prompt_messages, question=question)
        constructed_response_info = construct_reference(args.prompt_ver, ground_truth=ground_truth)

        if ground_truth is None or ground_truth == "":
            print(f"Warning: Ground truth is empty, got {ground_truth} from {messages}")
            return None
                
        data = {
            "data_source": f"sft_tulu3",
            "prompt": constructed_messages["messages"],
            "gen_prompt": constructed_messages["gen_messages"],
            **constructed_response_info,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "ability": "general",
            "extra_info": {
                "id": example.get("id", f"tulu3_{idx}"),
                "source": example.get("source", ""),
                "original_messages": messages,
            },
        }
        return data

    # Process the dataset
    train_dataset = train_dataset.map(function=process_fn, with_indices=True, num_proc=args.num_proc, load_from_cache_file=False)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True, num_proc=args.num_proc, load_from_cache_file=False)

    # Filter out None values
    train_dataset = train_dataset.filter(lambda x: x is not None, num_proc=args.num_proc)
    test_dataset = test_dataset.filter(lambda x: x is not None, num_proc=args.num_proc)
    print(f"Processed train dataset size: {len(train_dataset)}")
    print(f"Processed test dataset size: {len(test_dataset)}")

    # Filter out responses that are too long
    if args.max_response_length > 0:
        train_dataset = train_dataset.filter(lambda x: len(x["reference_content"]) <= args.max_response_length, num_proc=args.num_proc)
        test_dataset = test_dataset.filter(lambda x: len(x["reference_content"]) <= args.max_response_length, num_proc=args.num_proc)
        print(f"Filtered train dataset size: {len(train_dataset)}")
        print(f"Filtered test dataset size: {len(test_dataset)}")

    # Remove original columns
    train_dataset = train_dataset.remove_columns(["messages", "id", "source"])
    test_dataset = test_dataset.remove_columns(["messages", "id", "source"])

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset_path = os.path.join(args.output_dir, 'train.parquet')
    train_dataset.to_parquet(train_dataset_path)
    print(f"Saved train data to {train_dataset_path}")

    test_dataset_path = os.path.join(args.output_dir, 'test.parquet')
    test_dataset.to_parquet(test_dataset_path)
    print(f"Saved test data to {test_dataset_path}")

    if args.sample_size is not None:
        sampled_train_dataset = train_dataset.select(range(min(args.sample_size, len(train_dataset))))
        sampled_train_dataset_path = os.path.join(args.output_dir, f'train-n{args.sample_size}.parquet')
        sampled_train_dataset.to_parquet(sampled_train_dataset_path)
        print(f"Saved sampled train data to {sampled_train_dataset_path}")

        sampled_test_dataset = test_dataset.select(range(min(args.sample_size, len(test_dataset))))
        sampled_test_dataset_path = os.path.join(args.output_dir, f'test-n{args.sample_size}.parquet')
        sampled_test_dataset.to_parquet(sampled_test_dataset_path)
        print(f"Saved sampled test data to {sampled_test_dataset_path}")

