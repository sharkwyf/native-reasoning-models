#!/usr/bin/env python3
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
"""
Script to add special token "<|response_start|>" to a given model with proper initialization.
The embedding and lm_head are initialized from the eos_token.

Usage:
    python add_response_start_token_to_model.py --model_path <path_to_model> [--output_path <output_path>]

Example:
    python add_response_start_token_to_model.py --model_path ./meta-llama/Llama-3.2-3B --output_path ./modified/meta-llama/Llama-3.2-3B
    python add_response_start_token_to_model.py --model_path ./meta-llama/Llama-3.1-8B --output_path ./modified/meta-llama/Llama-3.1-8B
"""

import argparse
import json
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def initialize_new_token_embedding(model, tokenizer, token_id):
    """
    Initialize the embedding and output head for a new token using the eos_token's weights.

    For generation, both input embeddings and output head (lm_head) need proper initialization.
    - If weights are tied: Initializing input embedding automatically initializes output head
    - If weights are untied: Both need separate initialization using the same source (eos_token)

    Args:
        model: The model with resized embeddings
        tokenizer: The tokenizer to get eos_token_id
        token_id: The ID of the new token
    """
    # Get eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer does not have an eos_token_id. Cannot initialize from eos_token.")

    print(f"Initializing from eos_token (ID: {eos_token_id})")

    # Get the embedding layer
    if hasattr(model, "get_input_embeddings"):
        embedding_layer = model.get_input_embeddings()
    elif hasattr(model, "embed_tokens"):
        embedding_layer = model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")

    # Get the embedding weight
    embedding_weight = embedding_layer.weight.data

    # Get eos_token embedding
    eos_embedding = embedding_weight[eos_token_id].clone()

    # Set the new token embedding to match eos_token
    embedding_weight[token_id] = eos_embedding

    # Check if weights are tied
    # If tied, the output head shares weights with input embeddings, so no separate init needed
    # If untied, we need to initialize the output head row separately
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", None)

    # Get output embeddings/head
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
    elif hasattr(model, "lm_head"):
        output_embeddings = model.lm_head

    if output_embeddings is not None:
        output_weight = output_embeddings.weight.data

        # Check if weights are tied
        # Method 1: Check config flag
        # Method 2: Check if they share the same memory (same tensor)
        weights_are_tied = False

        if tie_word_embeddings is True:
            weights_are_tied = True
        elif output_weight.shape == embedding_weight.shape:
            # Check if they're the same tensor object
            try:
                # Try to check if they share memory
                if output_weight.data_ptr() == embedding_weight.data_ptr():
                    weights_are_tied = True
            except:
                # If comparison fails, assume untied
                pass

        if weights_are_tied:
            # Weights are tied, already initialized via input embedding
            print("  Output head weights are tied with input embeddings - using same initialization")
        else:
            # Weights are untied, need separate initialization for output head
            print("  Output head weights are untied - initializing separately from eos_token")

            # Get eos_token output head row
            eos_output_row = output_weight[eos_token_id].clone()

            # Set the new token output head row to match eos_token
            output_weight[token_id] = eos_output_row


def add_response_start_token(model_path, output_path=None, device="auto"):
    """
    Add the special token "<|response_start|>" to a model.
    The embedding and lm_head are initialized from the eos_token.

    Args:
        model_path: Path to the original model
        output_path: Path to save the modified model (default: models/modified/<model_name>)
        device: Device to load the model on ("auto", "cuda", "cpu")
    """
    print(f"Loading model from: {model_path}")

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # Define the special token
    response_start_token = "<|response_start|>"
    special_tokens = [response_start_token]

    # Get existing additional_special_tokens to preserve them
    existing_additional_tokens = []
    if hasattr(tokenizer, "additional_special_tokens"):
        existing_additional_tokens = list(tokenizer.additional_special_tokens)
    elif hasattr(tokenizer, "tokenizer_config"):
        # Try to get from tokenizer_config if available
        config = getattr(tokenizer, "tokenizer_config", {})
        if isinstance(config, dict) and "additional_special_tokens" in config:
            existing_additional_tokens = list(config["additional_special_tokens"])

    if len(existing_additional_tokens) > 0:
        print(f"Found {len(existing_additional_tokens)} existing additional special tokens: {existing_additional_tokens}")
    else:
        print("No existing additional special tokens found.")

    # Check which tokens already exist in the vocabulary
    tokens_to_add = []
    token_ids = {}
    vocab = tokenizer.get_vocab()

    for token in special_tokens:
        if token in vocab:
            print(f"Token '{token}' already exists in the tokenizer vocabulary!")
            token_ids[token] = tokenizer.convert_tokens_to_ids(token)
            print(f"  Token ID: {token_ids[token]}")
        else:
            tokens_to_add.append(token)

    # Merge existing tokens with new tokens, avoiding duplicates
    # We include ALL existing additional_special_tokens to preserve them in the config,
    # even if they're already in the vocab (add_special_tokens handles this gracefully)
    all_additional_tokens = list(existing_additional_tokens)

    # Add the new tokens we want to add (avoiding duplicates)
    for token in tokens_to_add:
        if token not in all_additional_tokens:
            all_additional_tokens.append(token)

    # Add the special tokens (preserving existing ones in the config)
    if tokens_to_add:
        print(f"Adding new special tokens: {tokens_to_add}")
        if len(existing_additional_tokens) > 0:
            print(f"Preserving existing additional special tokens: {existing_additional_tokens}")
        if len(all_additional_tokens) > len(tokens_to_add):
            print(f"All additional special tokens to be registered: {all_additional_tokens}")

        # Add all tokens (existing + new) to preserve the full list in tokenizer config
        # add_special_tokens will only actually add tokens that don't exist in vocab
        tokenizer.add_special_tokens({"additional_special_tokens": all_additional_tokens})
        new_vocab_size = len(tokenizer)
        print(f"New vocabulary size: {new_vocab_size}")

        # Get token IDs for all tokens
        for token in special_tokens:
            token_ids[token] = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token} -> Token ID: {token_ids[token]}")
    else:
        print("All target tokens already exist in the tokenizer.")
        for token in special_tokens:
            token_ids[token] = tokenizer.convert_tokens_to_ids(token)

        # Even if all target tokens exist, we should still ensure existing additional_special_tokens
        # are preserved in the tokenizer config when saving
        if len(existing_additional_tokens) > 0:
            # Re-register to ensure they're in the config (won't add duplicates to vocab)
            tokenizer.add_special_tokens({"additional_special_tokens": all_additional_tokens})
            print(f"Preserved existing additional special tokens in config: {existing_additional_tokens}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    # Resize token embeddings if needed
    if len(tokenizer) > original_vocab_size:
        print(f"Resizing model embeddings from {original_vocab_size} to {len(tokenizer)}...")
        model.resize_token_embeddings(len(tokenizer))

        # Initialize the new token embeddings from eos_token
        print("Initializing new token embeddings from eos_token...")
        for token in tokens_to_add:
            token_id = token_ids[token]
            print(f"  Initializing {token} (ID: {token_id})...")
            initialize_new_token_embedding(model, tokenizer, token_id)
        print("Token embeddings initialized successfully!")

    # Determine output path
    if output_path is None:
        # Extract model name from path
        model_name = os.path.basename(model_path.rstrip("/"))
        parent_dir = os.path.basename(os.path.dirname(model_path.rstrip("/")))
        # Create output path in models/modified
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, parent_dir, model_name)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save tokenizer and model
    print(f"\nSaving modified model and tokenizer to: {output_path}")
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    # Copy additional files that aren't part of standard HuggingFace save
    # These might include params.json, README.md, tokenizer.model.v3, etc.
    # Note: Model weight files (like consolidated.safetensors) are handled by save_pretrained
    print("Copying additional model files...")
    additional_files_to_copy = [
        "params.json",
        "README.md",
        # Note: tokenizer.model.v3 is typically identical to tokenizer.model
        # Since we're modifying the tokenizer, the .v3 file would be outdated
        # We skip it - if needed, it can be regenerated from the updated tokenizer.model
    ]

    for filename in additional_files_to_copy:
        src_path = os.path.join(model_path, filename)
        dst_path = os.path.join(output_path, filename)
        if os.path.exists(src_path):
            if filename == "params.json":
                # Update vocab_size in params.json if it exists
                try:
                    with open(src_path, "r") as f:
                        params = json.load(f)
                    if "vocab_size" in params and len(tokenizer) != original_vocab_size:
                        old_vocab_size = params["vocab_size"]
                        params["vocab_size"] = len(tokenizer)
                        print(f"  Updated params.json: vocab_size {old_vocab_size} -> {len(tokenizer)}")
                    with open(dst_path, "w") as f:
                        json.dump(params, f, indent=4)
                    print(f"  Copied and updated: {filename}")
                except Exception as e:
                    print(f"  Warning: Could not update {filename}: {e}")
                    # Fall back to simple copy
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied: {filename}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {filename}")

    # Also copy any other files that might be in the original directory
    # but weren't saved by save_pretrained (excluding model weights and standard HF files)
    standard_hf_files = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",  # tokenizer.model.v3 is handled separately above
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
        "model.safetensors",
        "flax_model.msgpack",
        "tf_model.h5",
    }

    # Copy other files that might exist
    if os.path.isdir(model_path):
        for item in os.listdir(model_path):
            src_item = os.path.join(model_path, item)
            dst_item = os.path.join(output_path, item)

            # Skip if already copied or is a standard HF file
            if os.path.exists(dst_item):
                continue

            # Skip model weight files (already saved by save_pretrained)
            # These are handled by save_pretrained, so we don't need to copy them
            if item.startswith("model-") and (item.endswith(".safetensors") or item.endswith(".bin")):
                continue
            if item in standard_hf_files:
                continue

            # Copy other files/directories
            if os.path.isfile(src_item):
                try:
                    shutil.copy2(src_item, dst_item)
                    print(f"  Copied additional file: {item}")
                except Exception as e:
                    print(f"  Warning: Could not copy {item}: {e}")

    print("\n✅ Successfully added special token '<|response_start|>' to the model!")
    print(f"   Model saved to: {output_path}")
    print(f"   Token ID: {response_start_token} -> {token_ids[response_start_token]}")
    print(f"   Vocabulary size: {len(tokenizer)}")

    # Verify the token
    print("\nVerifying token...")
    test_text = f"Test {response_start_token} token"
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    print(f"   Encoded and decoded: '{decoded}'")

    if response_start_token in decoded:
        print(f"   ✅ {response_start_token} verification successful!")
    else:
        print(
            f"   ⚠️  {response_start_token} verification: token may not appear in decoded text (this is normal for special tokens)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Add special token '<|response_start|>' to a model. The embedding and lm_head are initialized from the eos_token."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original model")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the modified model (default: models/modified/<model_name>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to load the model on (default: auto)",
    )

    args = parser.parse_args()

    add_response_start_token(model_path=args.model_path, output_path=args.output_path, device=args.device)


if __name__ == "__main__":
    main()

