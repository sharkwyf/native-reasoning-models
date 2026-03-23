# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_nrt_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger



# Define the Ray actor for parallel processing
@ray.remote(num_cpus=1)
class BatchItemProcessor:
    def __init__(self, tokenizer, config, should_create_baseline_batch):
        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.should_create_baseline_batch = should_create_baseline_batch
        
    def extract_thinking(self, text):
        """Extract thinking content before any special tokens
        """ 
        # if match any "<|.*|>" in text, extract the content before it
        match = re.search(r"<\|.*?\|>", text)
        if match:
            text = text[:match.start()]
            matched_text = match.group(0)
        else:
            matched_text = None
        
        return text, matched_text
        
    def process_item(
        self,
        prompt_ids,
        prompt_mask,
        response_ids,
        response_mask,
        think_prefix,
        response_prefix,
        reference_content,
    ):
        if self.config.algorithm.response_format == "null_think":
            generated_response = self.tokenizer.decode(response_ids[response_mask], skip_special_tokens=False)
            think_content = ""
            response_prefix = ""
            matched_text = ""
            is_response_prefix_matched = False

        elif self.config.algorithm.response_format == "response_prefix_as_think":
            generated_response = self.tokenizer.decode(response_ids[response_mask], skip_special_tokens=False)
            think_content, matched_text = self.extract_thinking(generated_response)
            is_response_prefix_matched = matched_text == response_prefix

            think_content = f"{think_content}{response_prefix}"
            response_prefix = ""
            
        elif self.config.algorithm.response_format == "default":
            generated_response = self.tokenizer.decode(response_ids[response_mask], skip_special_tokens=False)
            think_content, matched_text = self.extract_thinking(generated_response)
            is_response_prefix_matched = matched_text == response_prefix

        else:
            raise ValueError(f"Invalid response format: {self.config.algorithm.response_format}")
        
        null_think_content = ""
        reference_content = f"{reference_content}{self.tokenizer.eos_token}"

        # Tokenize the values of the dictionary in a batch
        components_dict = {
            "think_content": think_content,
            "null_think_content": null_think_content,
            "response_prefix": response_prefix,
            "reference_content": reference_content,
        }
        tokenized_output = self.tokenizer(
            list(components_dict.values()), padding=False, return_tensors=None, add_special_tokens=False
        )
        tokenized_input_ids = {
            key: np.array(tokenized_output['input_ids'][i], dtype=np.int64)
            for i, key in enumerate(components_dict.keys())
        }

        # --- 1. Structure data and apply truncation correctly ---
        # Organize components for easier processing. We only need the 'input_ids'.
        components = [
            tokenized_input_ids["think_content"],
            tokenized_input_ids["response_prefix"],
            tokenized_input_ids["reference_content"],
        ]
        # Correctly truncate the 'think_content' if the combined length exceeds the maximum.
        full_response_length = sum(len(c) for c in components)
        if full_response_length > self.max_response_length:
            # Calculate how much to remove.
            overflow = full_response_length - self.max_response_length
            # Truncate from the 'think_content' component, ensuring length is not negative.
            new_think_len = max(0, len(components[0]) - overflow)  
            components[0] = components[0][:new_think_len]

        # --- 2. Build final sequences and masks efficiently ---
        # Concatenate the final components to create the full response.
        processed_response_ids = np.concatenate(components)
        
        # An attention mask indicates which tokens the model should pay attention to.
        # Since we have no padding within this sequence, the mask is all ones.
        processed_response_mask = np.ones_like(processed_response_ids, dtype=np.int64)

        # --- 3. Create segment masks using a more performant approach ---
        # Get the lengths of the final components.
        lengths = np.array([len(c) for c in components], dtype=np.int32)
        total_len = lengths.sum()
        
        # Use cumulative sum to find start and end indices of each segment.
        # This is more efficient than repeatedly calling sum() inside a loop.
        endpoints = np.cumsum(lengths)
        startpoints = endpoints - lengths
        
        segment_masks = []
        for start, end in zip(startpoints, endpoints):
            mask = np.zeros(total_len, dtype=np.int64)
            mask[start:end] = 1
            segment_masks.append(mask)

        think_content_mask_padded, response_prefix_mask_padded, reference_content_mask_padded = segment_masks
        
        # ================= OPTIMIZED LOGIC ENDS HERE =================
        result = {
            "batch": {
                'prompt_ids': prompt_ids,
                'prompt_mask': prompt_mask,
                'processed_response_ids': processed_response_ids,
                'processed_response_mask': processed_response_mask,
                'think_content_mask': think_content_mask_padded,
                'response_prefix_mask': response_prefix_mask_padded,
                'reference_content_mask': reference_content_mask_padded,
                # for logging
                'prompt': self.tokenizer.decode(prompt_ids[prompt_mask == 1], skip_special_tokens=False),
                'generated_response': generated_response,
                'processed_response': self.tokenizer.decode(processed_response_ids, skip_special_tokens=False),
                'think_content': self.tokenizer.decode(processed_response_ids[think_content_mask_padded == 1], skip_special_tokens=False),
                'response_prefix': self.tokenizer.decode(processed_response_ids[response_prefix_mask_padded == 1], skip_special_tokens=False),
                'reference_content': self.tokenizer.decode(processed_response_ids[reference_content_mask_padded == 1], skip_special_tokens=False),
                "matched_think_eos_text": matched_text,
                'is_response_prefix_matched': is_response_prefix_matched,
            }
        }

        # process baseline batch (unchanged)
        if self.should_create_baseline_batch:
            # Use null think as a baseline. The components do not include any "think" tokens.
            baseline_components = [
                tokenized_input_ids["null_think_content"],
                tokenized_input_ids["response_prefix"], 
                tokenized_input_ids["reference_content"]
            ]
            
            # Concatenate components to create the final baseline sequence.
            baseline_response_ids = np.concatenate(baseline_components)
            baseline_response_mask = np.ones_like(baseline_response_ids, dtype=np.int64)
            
            # --- Create segment masks for the baseline batch ---
            baseline_lengths = np.array([len(c) for c in baseline_components], dtype=np.int32)
            total_baseline_len = baseline_lengths.sum()
            
            baseline_endpoints = np.cumsum(baseline_lengths)
            baseline_startpoints = baseline_endpoints - baseline_lengths

            baseline_segment_masks = []
            for start, end in zip(baseline_startpoints, baseline_endpoints):
                mask = np.zeros(total_baseline_len, dtype=np.int64)
                mask[start:end] = 1
                baseline_segment_masks.append(mask)

            baseline_think_content_mask_padded, baseline_response_prefix_mask_padded, baseline_reference_content_mask_padded = baseline_segment_masks
            
            result["baseline_batch"] = {
                'prompt_ids': prompt_ids,
                'prompt_mask': prompt_mask,
                'processed_response_ids': baseline_response_ids,
                'processed_response_mask': baseline_response_mask,
                'think_content_mask': baseline_think_content_mask_padded,
                'response_prefix_mask': baseline_response_prefix_mask_padded,
                'reference_content_mask': baseline_reference_content_mask_padded,
                # for logging
                'baseline_prompt': self.tokenizer.decode(prompt_ids[prompt_mask == 1], skip_special_tokens=False),
                "baseline_response": self.tokenizer.decode(baseline_response_ids[baseline_response_mask == 1], skip_special_tokens=False),
                'baseline_think_content': self.tokenizer.decode(baseline_response_ids[baseline_think_content_mask_padded == 1], skip_special_tokens=False),
                'baseline_response_prefix': self.tokenizer.decode(baseline_response_ids[baseline_response_prefix_mask_padded == 1], skip_special_tokens=False),
                'baseline_reference_content': self.tokenizer.decode(baseline_response_ids[baseline_reference_content_mask_padded == 1], skip_special_tokens=False),
            }

            if baseline_response_prefix_mask_padded.sum(-1) != response_prefix_mask_padded.sum(-1):
                raise Exception("baseline_response_prefix_mask_padded.sum(-1) != response_prefix_mask_padded.sum(-1)")

            if baseline_reference_content_mask_padded.sum(-1) != reference_content_mask_padded.sum(-1):
                raise Exception("baseline_reference_content_mask_padded.sum(-1) != reference_content_mask_padded.sum(-1)")
        
        return result


class RayNRTTrainer(RayPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a pool of actors
        # self.should_create_baseline_batch = self.config.algorithm.use_clipped_reward or \
        #     (self.config.algorithm.reward_agg_mode in ["weighted_sum_inverse_prob", "weighted_sum_log_prob"])
        self.should_create_baseline_batch = True

        from ray.util.actor_pool import ActorPool
        self.num_actors = 4
        self.actors = [BatchItemProcessor.remote(self.tokenizer, self.config, self.should_create_baseline_batch) for _ in range(self.num_actors)]
        self.actor_pool = ActorPool(self.actors)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        """
        data_item.batch
        TensorDict(
            fields={
                advantages: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                attention_mask: Tensor(shape=torch.Size([5818]), device=cpu, dtype=torch.int64, is_shared=False),
                clipped_token_reward: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                clipped_trace_reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                entropys: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                input_ids: Tensor(shape=torch.Size([5818]), device=cpu, dtype=torch.int64, is_shared=False),
                is_clipped: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                is_response_prefix_matched: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                normalized_token_reward: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                normalized_trace_reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                old_log_probs: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                position_ids: Tensor(shape=torch.Size([5818]), device=cpu, dtype=torch.int64, is_shared=False),
                prompts: Tensor(shape=torch.Size([2048]), device=cpu, dtype=torch.int64, is_shared=False),
                raw_token_reward: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                raw_trace_reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reference_content_masks: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.int64, is_shared=False),
                response_mask: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.int64, is_shared=False),
                response_prefix_masks: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.int64, is_shared=False),
                responses: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.int64, is_shared=False),
                returns: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                think_content_masks: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.int64, is_shared=False),
                token_level_rewards: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                token_level_scores: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                token_reward: Tensor(shape=torch.Size([3770]), device=cpu, dtype=torch.float32, is_shared=False),
                trace_reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

        data_item.non_tensor_batch.keys()
        dict_keys(['data_source', 'reward_model', 'extra_info', 'uid', 'interaction_kwargs', 'think_prefix', 'response_prefix', 'gen_prompt', 'ability', 'reference_content', 'tools_kwargs', 'index', 'prompt', 'processed_response', 'generated_response', 'think_content', 'matched_think_eos_text', 'baseline_prompt', 'baseline_response', 'baseline_think_content', 'baseline_response_prefix', 'baseline_reference_content', 'score'])
        """
        all_results = []

        for i, data_item in enumerate(batch):
            all_results.append({
                "prompt": data_item.non_tensor_batch["prompt"],
                "response": data_item.non_tensor_batch["processed_response"],
                "think_content_length": data_item.batch["think_content_masks"].sum(),
                "think_content": data_item.non_tensor_batch["think_content"],
                "response_prefix": data_item.non_tensor_batch["response_prefix"],
                "reference_content": data_item.non_tensor_batch["reference_content"],

                "generated_response": data_item.non_tensor_batch["generated_response"],
                "matched_think_eos_text": data_item.non_tensor_batch["matched_think_eos_text"],
                "is_response_prefix_matched": data_item.batch["is_response_prefix_matched"],

                "reward": {
                    "trace_reward": data_item.batch["trace_reward"],
                    "raw_trace_reward": data_item.batch["raw_trace_reward"],
                    "scaled_trace_reward": data_item.batch["scaled_trace_reward"],
                },

                "baseline_prompt": data_item.non_tensor_batch["baseline_prompt"],
                "baseline_response": data_item.non_tensor_batch["baseline_response"],
                "baseline_think_content": data_item.non_tensor_batch["baseline_think_content"],
                "baseline_response_prefix": data_item.non_tensor_batch["baseline_response_prefix"],
                "baseline_reference_content": data_item.non_tensor_batch["baseline_reference_content"],
                "raw_item": {
                    "data_source": data_item.non_tensor_batch["data_source"],
                    "reward_model": data_item.non_tensor_batch["reward_model"],
                    "extra_info": data_item.non_tensor_batch["extra_info"],
                    "uid": data_item.non_tensor_batch["uid"],
                },
            })

        def _serialize_item(item):
            """Recursively serialize an item (dict, list, etc.) to ensure JSON compatibility."""
            if item is None:
                return None
            elif isinstance(item, dict):
                return {k: _serialize_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [_serialize_item(v) for v in item]
            elif isinstance(item, torch.Tensor):
                return _serialize_item(item.cpu().numpy().tolist())
            elif isinstance(item, np.ndarray):
                return _serialize_item(item.tolist())
            elif isinstance(item, bytes):
                return None
            elif isinstance(item, str) or isinstance(item, int) or isinstance(item, float):
                return item
            else:
                raise Exception(f"Unsupported data type {type(item)} in item: {item}")
        
        all_results = _serialize_item(all_results)
        all_results = sorted(all_results, key=lambda x: x['prompt'], reverse=True)

        # Save all_results as a JSON file
        os.makedirs(rollout_data_dir, exist_ok=True)
        save_path = os.path.join(rollout_data_dir, f"rollouts-step-{self.global_steps}.json")
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            print(f"Saved rollouts to {save_path}")


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", other_batchs=None):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

        for other_batch in other_batchs or []:
            if isinstance(other_batch, DataProto):
                other_batch.reorder(global_idx)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Added: combine train batch for answer log probs calculation
                    with marked_timer('combine_train_batch', timing_raw):
                        batch, baseline_batch = self.combine_train_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics, other_batchs=[baseline_batch])

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # compute answer_log_prob
                    with marked_timer('calculate_nrt_rewards', timing_raw):
                        batch = self.calculate_answer_reward(batch, baseline_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_nrt_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


    def combine_train_batch(self, batch, gen_batch_output):
        """
        Concat input for answer log probs calculation with Ray parallelization
        """
        timing_raw = {}
        baseline_batch = None

        with marked_timer('combine_train_batch', timing_raw):
            batch_size = len(batch)

            # Initialize batch data
            with marked_timer('initialize_batch_data', timing_raw):
                prompt_ids = batch.batch["prompts"].cpu().numpy()
                prompt_lens = batch.batch["prompts"].shape[-1]
                prompt_attention_mask = (batch.batch["attention_mask"][:, :prompt_lens] == 1).cpu().numpy()
                response_ids = gen_batch_output.batch["responses"].cpu().numpy()
                response_attention_mask = (compute_response_mask(gen_batch_output) == 1).cpu().numpy()
                
                think_prefix_texts = batch.non_tensor_batch["think_prefix"]
                response_prefix_texts = batch.non_tensor_batch["response_prefix"]
                reference_content = batch.non_tensor_batch["reference_content"]
            
            # Create tasks for parallel processing
            with marked_timer('create_tasks', timing_raw):
                tasks = []
                for i in range(batch_size):
                    tasks.append({
                        "prompt_ids": prompt_ids[i],
                        "prompt_mask": prompt_attention_mask[i],
                        "response_ids": response_ids[i],
                        "response_mask": response_attention_mask[i],
                        "think_prefix": think_prefix_texts[i],
                        "response_prefix": response_prefix_texts[i],
                        "reference_content": reference_content[i],
                    })

            # Process in parallel
            with marked_timer('process_in_parallel', timing_raw):
                results = list(self.actor_pool.map(lambda actor, kwargs: actor.process_item.remote(**kwargs), tasks))
            
            # Collect results
            with marked_timer('collect_results', timing_raw):
                all_prompt_ids = []
                all_prompt_masks = []
                all_processed_response_ids = []
                all_processed_response_masks = []
                all_think_content_masks = []
                all_response_prefix_masks = []
                all_reference_content_masks = []
                all_prompt_texts = []
                all_generated_responses = []
                all_processed_responses = []
                all_think_content_texts = []
                all_response_prefix_texts = []
                all_reference_content_texts = []
                all_matched_think_eos_text = []
                all_is_response_prefix_matched = []
                
                for result in results:
                    result_batch = result['batch']
                    all_prompt_ids.append(result_batch['prompt_ids'])
                    all_prompt_masks.append(result_batch['prompt_mask'])
                    all_processed_response_ids.append(result_batch['processed_response_ids'])
                    all_processed_response_masks.append(result_batch['processed_response_mask'])
                    all_think_content_masks.append(result_batch['think_content_mask'])
                    all_response_prefix_masks.append(result_batch['response_prefix_mask'])
                    all_reference_content_masks.append(result_batch['reference_content_mask'])

                    all_prompt_texts.append(result_batch['prompt'])
                    all_generated_responses.append(result_batch['generated_response'])
                    all_processed_responses.append(result_batch['processed_response'])
                    all_think_content_texts.append(result_batch['think_content'])
                    all_response_prefix_texts.append(result_batch['response_prefix'])
                    all_reference_content_texts.append(result_batch['reference_content'])
                    all_matched_think_eos_text.append(result_batch['matched_think_eos_text'])
                    all_is_response_prefix_matched.append(result_batch['is_response_prefix_matched'])

            # Pad sequences to the same length
            with marked_timer('pad_sequences', timing_raw):
                max_prompt_len = min(max([len(ids) for ids in all_prompt_ids]), self.config.data.max_prompt_length)
                max_response_len = self.config.data.max_response_length + self.config.data.max_think_length
                max_response_len = min(max([len(ids) for ids in all_processed_response_ids]), max_response_len)
                pad_token_id = self.tokenizer.pad_token_id
            
                # Create padded arrays with the correct size
                padded_prompt_ids = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
                padded_prompt_masks = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
                padded_processed_response_ids = np.zeros((batch_size, max_response_len), dtype=np.int64)
                padded_processed_response_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                padded_think_content_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                padded_response_prefix_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                padded_reference_content_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                
                # Fill in the actual values
                for i in range(batch_size):
                    prompt_len = min(len(all_prompt_ids[i]), max_prompt_len)
                    response_len = min(len(all_processed_response_ids[i]), max_response_len)
                    padded_prompt_ids[i, -prompt_len:] = all_prompt_ids[i][-prompt_len:]
                    padded_prompt_masks[i, -prompt_len:] = all_prompt_masks[i][-prompt_len:]
                    padded_processed_response_ids[i, :response_len] = all_processed_response_ids[i][:response_len]
                    padded_processed_response_masks[i, :response_len] = all_processed_response_masks[i][:response_len]
                    padded_think_content_masks[i, :response_len] = all_think_content_masks[i][:response_len]
                    padded_response_prefix_masks[i, :response_len] = all_response_prefix_masks[i][:response_len]
                    padded_reference_content_masks[i, :response_len] = all_reference_content_masks[i][:response_len]
                    
                    # Set padding token for input_ids
                    if prompt_len < max_prompt_len:
                        padded_prompt_ids[i, :-prompt_len] = pad_token_id
                    if response_len < max_response_len:
                        padded_processed_response_ids[i, response_len:] = pad_token_id

            # Convert all the padded arrays to tensors
            with marked_timer('convert_to_tensors', timing_raw):
                all_prompt_ids = torch.from_numpy(padded_prompt_ids)
                all_prompt_masks = torch.from_numpy(padded_prompt_masks)
                all_processed_response_ids = torch.from_numpy(padded_processed_response_ids)
                all_processed_response_masks = torch.from_numpy(padded_processed_response_masks)
                all_think_content_masks = torch.from_numpy(padded_think_content_masks)
                all_response_prefix_masks = torch.from_numpy(padded_response_prefix_masks)
                all_reference_content_masks = torch.from_numpy(padded_reference_content_masks)
                all_is_response_prefix_matched = torch.from_numpy(np.array(all_is_response_prefix_matched))

            # Create final batch tensors
            with marked_timer('create_final_batch_tensors', timing_raw):
                input_ids = torch.cat((all_prompt_ids, all_processed_response_ids), dim=-1)
                attention_mask = torch.cat((all_prompt_masks, all_processed_response_masks), dim=-1)
                position_ids = compute_position_id_with_mask(attention_mask)
            
            # Update batch
            with marked_timer('update_batch', timing_raw):
                batch.batch['input_ids'] = input_ids
                batch.batch['attention_mask'] = attention_mask
                batch.batch['position_ids'] = position_ids
                batch.batch['prompts'] = all_prompt_ids
                batch.batch['responses'] = all_processed_response_ids
                batch.batch['response_mask'] = all_think_content_masks | all_response_prefix_masks
                batch.batch['think_content_masks'] = all_think_content_masks
                batch.batch['response_prefix_masks'] = all_response_prefix_masks
                batch.batch['reference_content_masks'] = all_reference_content_masks
                batch.batch['is_response_prefix_matched'] = all_is_response_prefix_matched
                batch.non_tensor_batch['prompt'] = np.array(all_prompt_texts)
                batch.non_tensor_batch['processed_response'] = np.array(all_processed_responses)
                batch.non_tensor_batch['generated_response'] = np.array(all_generated_responses)
                batch.non_tensor_batch['think_content'] = np.array(all_think_content_texts)
                batch.non_tensor_batch['response_prefix'] = np.array(all_response_prefix_texts)
                batch.non_tensor_batch['reference_content'] = np.array(all_reference_content_texts)
                batch.non_tensor_batch['matched_think_eos_text'] = np.array(all_matched_think_eos_text)

            # Create baseline answer batch
            if self.should_create_baseline_batch:
                with marked_timer('create_baseline_batch', timing_raw):
                    baseline_prompt_ids = []
                    baseline_prompt_masks = []
                    baseline_response_ids = []
                    baseline_response_masks = []
                    baseline_think_content_masks = []
                    baseline_response_prefix_masks = []
                    baseline_reference_content_masks = []
                    baseline_prompt_texts = []
                    baseline_response_texts = []
                    baseline_think_content_texts = []
                    baseline_response_prefix_texts = []
                    baseline_reference_content_texts = []
                    
                    for result in results:
                        result_batch = result['baseline_batch']
                        baseline_prompt_ids.append(result_batch["prompt_ids"])
                        baseline_prompt_masks.append(result_batch["prompt_mask"])
                        baseline_response_ids.append(result_batch["processed_response_ids"])
                        baseline_response_masks.append(result_batch["processed_response_mask"])
                        baseline_think_content_masks.append(result_batch["think_content_mask"])
                        baseline_response_prefix_masks.append(result_batch["response_prefix_mask"])
                        baseline_reference_content_masks.append(result_batch["reference_content_mask"])

                        baseline_prompt_texts.append(result_batch["baseline_prompt"])
                        baseline_response_texts.append(result_batch["baseline_response"])
                        baseline_think_content_texts.append(result_batch["baseline_think_content"])
                        baseline_response_prefix_texts.append(result_batch["baseline_response_prefix"])
                        baseline_reference_content_texts.append(result_batch["baseline_reference_content"])

                    max_prompt_len = min(max([len(ids) for ids in baseline_prompt_ids]), self.config.data.max_prompt_length)
                    max_response_len = min(max([len(ids) for ids in baseline_response_ids]), self.config.data.max_response_length)

                    # Create padded arrays with the correct size
                    padded_baseline_prompt_ids = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
                    padded_baseline_prompt_masks = np.zeros((batch_size, max_prompt_len), dtype=np.int64)
                    padded_baseline_response_ids = np.zeros((batch_size, max_response_len), dtype=np.int64)
                    padded_baseline_response_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                    padded_baseline_think_content_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                    padded_baseline_response_prefix_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)
                    padded_baseline_reference_content_masks = np.zeros((batch_size, max_response_len), dtype=np.int64)

                    # Fill in the actual values
                    for i in range(batch_size):
                        prompt_len = min(len(baseline_prompt_ids[i]), max_prompt_len)
                        response_len = min(len(baseline_response_ids[i]), max_response_len)

                        padded_baseline_prompt_ids[i, -prompt_len:] = baseline_prompt_ids[i][-prompt_len:]
                        padded_baseline_prompt_masks[i, -prompt_len:] = baseline_prompt_masks[i][-prompt_len:]
                        padded_baseline_response_ids[i, :response_len] = baseline_response_ids[i][:response_len]
                        padded_baseline_response_masks[i, :response_len] = baseline_response_masks[i][:response_len]
                        padded_baseline_think_content_masks[i, :response_len] = baseline_think_content_masks[i][:response_len]
                        padded_baseline_response_prefix_masks[i, :response_len] = baseline_response_prefix_masks[i][:response_len]
                        padded_baseline_reference_content_masks[i, :response_len] = baseline_reference_content_masks[i][:response_len]

                        # Set padding token for input_ids
                        if prompt_len < max_prompt_len:
                            padded_baseline_prompt_ids[i, :-prompt_len] = pad_token_id
                        if response_len < max_response_len:
                            padded_baseline_response_ids[i, response_len:] = pad_token_id

                    baseline_input_ids = torch.from_numpy(np.concatenate((padded_baseline_prompt_ids, padded_baseline_response_ids), axis=-1))
                    baseline_attention_mask = torch.from_numpy(np.concatenate((padded_baseline_prompt_masks, padded_baseline_response_masks), axis=-1))
                    baseline_position_ids = compute_position_id_with_mask(baseline_attention_mask)
                
                    padded_baseline_response_ids = torch.from_numpy(padded_baseline_response_ids)
                    padded_baseline_response_masks = torch.from_numpy(padded_baseline_response_masks)
                    padded_baseline_think_content_masks = torch.from_numpy(padded_baseline_think_content_masks)
                    padded_baseline_response_prefix_masks = torch.from_numpy(padded_baseline_response_prefix_masks)
                    padded_baseline_reference_content_masks = torch.from_numpy(padded_baseline_reference_content_masks)

                    baseline_batch = DataProto.from_dict({
                        "input_ids": baseline_input_ids,
                        "attention_mask": baseline_attention_mask,
                        "position_ids": baseline_position_ids,
                        "prompts": padded_baseline_prompt_ids,
                        "responses": padded_baseline_response_ids,
                        # "response_mask": padded_baseline_response_masks,
                        "think_content_masks": padded_baseline_think_content_masks,
                        "response_prefix_masks": padded_baseline_response_prefix_masks,
                        "reference_content_masks": padded_baseline_reference_content_masks,
                    })
                    batch.non_tensor_batch["baseline_prompt"] = np.array(baseline_prompt_texts)
                    batch.non_tensor_batch["baseline_response"] = np.array(baseline_response_texts)
                    batch.non_tensor_batch["baseline_think_content"] = np.array(baseline_think_content_texts)
                    batch.non_tensor_batch["baseline_response_prefix"] = np.array(baseline_response_prefix_texts)
                    batch.non_tensor_batch["baseline_reference_content"] = np.array(baseline_reference_content_texts)

        # print the timing for the entire function, and percentage of time of each step,
        # as well as the time taken for each step
        total_time = timing_raw['combine_train_batch']
        print(f"Total time for combine_train_batch: {total_time:.4f} seconds")
        for step, time_taken in timing_raw.items():
            percentage = (time_taken / total_time) * 100
            print(f"  {step}: {time_taken:.4f} seconds ({percentage:.2f}%)")

        return batch, baseline_batch

    def _compute_nrt_rewards(self, batch: DataProto, baseline_batch: DataProto, reward_agg_mode: str):
        """
        Compute the NRT reward.
        """
        raw_answer_log_prob = batch.batch["old_log_probs"]
        raw_answer_entropy = batch.batch["entropys"]
        raw_answer_mask = batch.batch["reference_content_masks"].float()
        
        T = raw_answer_mask.sum(dim=1).float()
        T_safe = T.clamp(min=1.0)

        if reward_agg_mode == "log_prob":
            # trace_reward = $\log \pi_\theta(y^\star|x,z)$
            # Vectorized: Sum log_probs where mask is 1
            trace_reward = (raw_answer_log_prob * raw_answer_mask).sum(dim=1)
            # token_reward = 1 for each valid token
            token_reward = raw_answer_mask

        elif reward_agg_mode == "prob":
            # trace_reward = $\pi_\theta(y^\star|x,z)$ (calculated in log-space for stability)
            trace_reward = torch.exp((raw_answer_log_prob * raw_answer_mask).sum(dim=1))
            # token_reward = $\pi_\theta(y^\star|x,z)$ (same value for all tokens)
            token_reward = trace_reward.unsqueeze(-1) * raw_answer_mask

        elif reward_agg_mode == "geometric_mean":
            # trace_reward = $(\pi_\theta(y^\star|x,z))^{1/T}$
            # Geometric mean in log-space: exp((1/T) * sum(log_probs))
            log_prob_sum = (raw_answer_log_prob * raw_answer_mask).sum(dim=1)
            trace_reward = torch.exp(log_prob_sum / T_safe)
            # token_reward = $\frac{1}{T}(\pi_\theta(y^\star|x,z))^{1/T}$
            token_reward = (trace_reward / T_safe).unsqueeze(-1) * raw_answer_mask

        elif reward_agg_mode == "arithmetic_mean":
            # trace_reward = $\frac{1}{T}\sum_{j=1}^T c_j$
            # Use log-sum-exp trick for numerical stability: log(mean(exp(x))) = logsumexp(x) - log(T)
            masked_log_prob = raw_answer_log_prob.masked_fill(raw_answer_mask == 0, -torch.inf)
            log_mean_prob = torch.logsumexp(masked_log_prob, dim=1) - torch.log(T_safe)
            trace_reward = torch.exp(log_mean_prob)
            # token_reward = $\frac{1}{T}c_i$
            token_reward = (torch.exp(raw_answer_log_prob) / T_safe.unsqueeze(-1)) * raw_answer_mask

        elif reward_agg_mode in ["weighted_sum_inverse_prob", "weighted_sum_log_prob"]:
            # Inverse-Probability: w_i ∝ 1/c_{i,base}, trace_reward = 1/T ∑c_j/c_{j,base}
            # Log-Probability: w_i ∝ -log c_{i,base}, trace_reward = -1/T ∑c_j log c_{j,base}
            baseline_log_prob = baseline_batch.batch["old_log_probs"]
            baseline_answer_mask = baseline_batch.batch["reference_content_masks"]
            batch_size = raw_answer_log_prob.shape[0]
            trace_reward = torch.zeros(batch_size, device=raw_answer_log_prob.device)
            token_reward = torch.zeros_like(raw_answer_log_prob)
            # Loop over each sample in the batch to handle potentially different mask lengths.
            for i in range(batch_size):
                # Find the indices of valid (non-masked) tokens for both sequences.
                raw_valid_indices = torch.where(raw_answer_mask[i] == 1)[0]
                baseline_valid_indices = torch.where(baseline_answer_mask[i] == 1)[0]
                # Determine the alignment length (truncate to the shorter sequence).
                min_len = min(len(raw_valid_indices), len(baseline_valid_indices))
                if min_len > 0:
                    # Get the indices of the first `min_len` valid tokens.
                    aligned_raw_indices = raw_valid_indices[:min_len]
                    aligned_baseline_indices = baseline_valid_indices[:min_len]
                    # Extract the corresponding log probabilities for the aligned tokens.
                    aligned_raw_log_prob = raw_answer_log_prob[i][aligned_raw_indices]
                    aligned_baseline_log_prob = baseline_log_prob[i][aligned_baseline_indices]
                    # Compute weights in log-space based on the baseline model's probabilities.
                    if reward_agg_mode == "weighted_sum_inverse_prob":
                        # log(w_j) = -log(c_{j,base})
                        weights = 1.0 / torch.exp(aligned_baseline_log_prob)
                    else:  # "weighted_sum_log_prob"
                        # log(w_j) = log(-log(c_{j,base}))
                        weights = - aligned_baseline_log_prob
                    
                    # Calculate weighted log probabilities: log(c_j * w_j) = log(c_j) + log(w_j)
                    weighted_log_probs = aligned_raw_log_prob + torch.log(weights)
                    
                    # Calculate trace reward: mean(c_j * w_j)
                    # Use log-sum-exp for stable sum, then convert to mean in normal space.
                    log_sum_weighted = torch.logsumexp(weighted_log_probs, dim=0)
                    trace_reward[i] = torch.exp(log_sum_weighted) / min_len
                    
                    # Calculate token rewards: (c_j * w_j) / min_len
                    # Place the calculated rewards back into the original tensor shape at the correct positions.
                    aligned_token_rewards = torch.exp(weighted_log_probs) / min_len
                    token_reward[i, aligned_raw_indices] = aligned_token_rewards

        else:
            raise ValueError(f"Invalid reward aggregation mode: {reward_agg_mode}")

        # check if nan or inf in trace_reward or token_reward
        if torch.isnan(trace_reward).any() or torch.isinf(trace_reward).any():
            raise Exception("trace_reward is nan or inf")
        if torch.isnan(token_reward).any() or torch.isinf(token_reward).any():
            raise Exception("token_reward is nan or inf")

        return trace_reward, token_reward

    @torch.no_grad()
    def calculate_answer_reward(self, batch: DataProto, baseline_batch: DataProto):
        """
        Calculate the answer log probability and reward information.
        This function computes the answer log probability and reward information based on the provided batch and answer_batch.
        It handles different answer formats and computes various metrics such as log probability reward, mean reward, and normalized reward.
        """
        if baseline_batch is not None:
            baseline_old_log_prob = self.actor_rollout_wg.compute_log_prob(baseline_batch)
            baseline_batch = baseline_batch.union(baseline_old_log_prob)

        trace_reward, token_reward = self._compute_nrt_rewards(batch, baseline_batch, self.config.algorithm.reward_agg_mode)
        
        batch.batch["raw_trace_reward"] = trace_reward.clone()
        batch.batch["raw_token_reward"] = token_reward.clone()

        if self.config.algorithm.center_reward_by == "mean":
            # Center trace_reward and token_reward separately by their respective means
            # Group rewards by unique ID (prompt) to calculate sigma per group
            index_uid = batch.non_tensor_batch['uid']
            id2trace_rewards = defaultdict(list)
            id2token_rewards = defaultdict(list)
            id2token_masks = defaultdict(list)
            id2indices = defaultdict(list)
            centered_trace_reward = torch.zeros_like(trace_reward)
            centered_token_reward = torch.zeros_like(token_reward)
            
            for i in range(len(trace_reward)):
                id2trace_rewards[index_uid[i]].append(trace_reward[i])
                id2token_rewards[index_uid[i]].append(token_reward[i])
                id2token_masks[index_uid[i]].append(batch.batch["reference_content_masks"][i])
                id2indices[index_uid[i]].append(i)
            
            # Calculate mean and apply normalization for each group
            for uid in id2trace_rewards:
                group_trace_rewards_tensor = torch.stack(id2trace_rewards[uid])
                group_token_rewards_tensor = torch.stack(id2token_rewards[uid])
                group_token_masks_tensor = torch.stack(id2token_masks[uid])
                group_indices = id2indices[uid]
                
                # Calculate mean for trace and token rewards
                trace_mean = torch.mean(group_trace_rewards_tensor)
                
                # Flatten and filter out masked tokens (where mask=0)
                valid_token_rewards = group_token_rewards_tensor[group_token_masks_tensor.bool()].flatten()
                token_mean = torch.mean(valid_token_rewards)
                                
                # Apply normalization to all samples in this group
                for original_idx in group_indices:
                    centered_trace_reward[original_idx] = trace_reward[original_idx] - trace_mean
                    centered_token_reward[original_idx] = token_reward[original_idx] - token_mean

            batch.batch["centered_trace_reward"] = centered_trace_reward.clone()
            batch.batch["centered_token_reward"] = centered_token_reward.clone()

            trace_reward = centered_trace_reward
            token_reward = centered_token_reward
        
        elif self.config.algorithm.center_reward_by is not None:
            raise ValueError(f"Invalid center_reward_by: {self.config.algorithm.center_reward_by}")

        if self.config.algorithm.scale_reward_by == "std":
            # Scale trace_reward and token_reward separately by their respective stds
            # Group rewards by unique ID (prompt) to calculate sigma per group
            index_uid = batch.non_tensor_batch['uid']
            id2trace_rewards = defaultdict(list)
            id2token_rewards = defaultdict(list)
            id2token_masks = defaultdict(list)
            id2indices = defaultdict(list)
            scaled_trace_reward = torch.zeros_like(trace_reward)
            scaled_token_reward = torch.zeros_like(token_reward)
            
            for i in range(len(trace_reward)):
                id2trace_rewards[index_uid[i]].append(trace_reward[i])
                id2token_rewards[index_uid[i]].append(token_reward[i])
                id2token_masks[index_uid[i]].append(batch.batch["reference_content_masks"][i])
                id2indices[index_uid[i]].append(i)
            
            # Calculate std and apply normalization for each group
            for uid in id2trace_rewards:
                group_trace_rewards_tensor = torch.stack(id2trace_rewards[uid])
                group_token_rewards_tensor = torch.stack(id2token_rewards[uid])
                group_token_masks_tensor = torch.stack(id2token_masks[uid])
                group_indices = id2indices[uid]
                
                # Calculate separate stds for trace and token rewards
                trace_std = torch.std(group_trace_rewards_tensor)
                
                # Flatten and filter out masked tokens (where mask=0)
                valid_token_rewards = group_token_rewards_tensor[group_token_masks_tensor.bool()].flatten()
                token_std = torch.std(valid_token_rewards)
                                
                # Apply normalization to all samples in this group
                for original_idx in group_indices:
                    if trace_std == 0.0:
                        scaled_trace_reward[original_idx] = 0
                    else:
                        scaled_trace_reward[original_idx] = trace_reward[original_idx] / trace_std

                    if token_std == 0.0:
                        scaled_token_reward[original_idx] = 0
                    else:
                        scaled_token_reward[original_idx] = token_reward[original_idx] / token_std

            batch.batch["scaled_trace_reward"] = scaled_trace_reward.clone()
            batch.batch["scaled_token_reward"] = scaled_token_reward.clone()

            trace_reward = scaled_trace_reward
            token_reward = scaled_token_reward

        elif self.config.algorithm.scale_reward_by is not None:
            raise ValueError(f"Invalid scale_reward_by: {self.config.algorithm.scale_reward_by}")

        if self.config.algorithm.use_fixed_token_reward:
            token_reward = batch.batch["reference_content_masks"].float()
        
        batch.batch["trace_reward"] = trace_reward
        batch.batch["token_reward"] = token_reward

        # check if nan or inf in trace_reward or token_reward
        if torch.isnan(trace_reward).any() or torch.isinf(trace_reward).any():
            print("trace_reward is nan or inf")
        if torch.isnan(token_reward).any() or torch.isinf(token_reward).any():
            print("token_reward is nan or inf")

        return batch

