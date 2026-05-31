#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:17:50
@Author  :   wangjiakang
@Modified:   2026/03/06 - V8 使用vllm获取next_token_embedding作为表征
@File    :   dapo_ray_trainer.py
'''


import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import json
import os
import datetime

import numpy as np
import math
import torch
from tqdm import tqdm

from dapo.dynamic_category_sampler import GlobalCategoryPoolSampler
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from typing import Dict, List, Any, Optional


class TrainingStateLogger:
    """
    V8: 训练状态日志记录器
    支持详细的训练状态输出和JSON格式存储
    """
    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []
        self.json_file_path = os.path.join(output_dir, f"{experiment_name}_training_state.json")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def log_step(self, step: int, metrics: Dict[str, Any], print_detail: bool = True):
        """
        记录单个step的训练状态

        Args:
            step: 当前步数
            metrics: 训练指标字典
            print_detail: 是否打印详细信息
        """
        # 构建状态记录
        state_record = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": self._serialize_metrics(metrics)
        }

        self.history.append(state_record)

        # 打印详细信息
        if print_detail:
            self._print_detailed_state(step, metrics)

        # 保存到JSON文件
        self._save_to_json()

    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """将metrics中的tensor/numpy转换为Python原生类型"""
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serialized[key] = value.item() if value.size == 1 else value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                serialized[key] = value
        return serialized

    def _print_detailed_state(self, step: int, metrics: Dict[str, Any]):
        """打印详细的训练状态"""
        print(f"\n{'='*70}")
        print(f"[Step {step}] Training State - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # 1. Reward相关指标
        self._print_category("Rewards by Domain", metrics, [
            ("batch/average_math_score", "Math Reward"),
            ("batch/average_code_score", "Code Reward"),
            ("batch/average_general_score", "General Reward"),
        ])

        # 2. Model/Rule Reward分解
        self._print_category("Reward Breakdown (Model/Rule)", metrics, [
            ("batch/average_math_model_score", "Math Model"),
            ("batch/average_math_rule_score", "Math Rule"),
            ("batch/average_code_model_score", "Code Model"),
            ("batch/average_code_rule_score", "Code Rule"),
            ("batch/average_general_model_score", "General Model"),
            ("batch/average_general_rule_score", "General Rule"),
        ])

        # 3. Batch统计
        self._print_category("Batch Statistics", metrics, [
            ("batch/valid", "Valid Samples"),
            ("batch/solve_none", "Solve None"),
            ("batch/solve_all", "Solve All"),
            ("batch/clip_overlong", "Clip Overlong"),
            ("batch/inverse_pair", "Inverse Pairs"),
        ])

        # 4. Loss相关
        self._print_category("Loss Metrics", metrics, [
            ("actor/entropy", "Actor Entropy"),
            ("critic/values", "Critic Values"),
            ("critic/loss", "Critic Loss"),
            ("actor/loss", "Actor Loss"),
            ("actor/pg_loss", "Actor PG Loss"),
            ("actor/kl_loss", "Actor KL Loss"),
        ])

        # 5. Dynamic Sampling相关
        self._print_category("Dynamic Sampling", metrics, [
            ("dynamic/weight_math", "Weight Math"),
            ("dynamic/weight_code", "Weight Code"),
            ("dynamic/weight_general", "Weight General"),
            ("dynamic/improvement_math", "Improvement Math"),
            ("dynamic/improvement_code", "Improvement Code"),
            ("dynamic/improvement_general", "Improvement General"),
            ("dynamic/distance_math", "Distance Math"),
            ("dynamic/distance_code", "Distance Code"),
            ("dynamic/distance_general", "Distance General"),
            ("dynamic/beta", "Beta"),
            ("dynamic/gamma", "Gamma"),
        ])

        # 6. Timing相关
        self._print_category("Timing (seconds)", metrics, [
            ("timing/gen", "Generation"),
            ("timing/reward", "Reward"),
            ("timing/old_log_prob", "Old Log Prob"),
            ("timing/update_actor", "Update Actor"),
            ("timing/update_critic", "Update Critic"),
            ("timing/step", "Total Step"),
        ])

        # 7. Throughput
        self._print_category("Throughput", metrics, [
            ("throughput/tokens_per_second", "Tokens/sec"),
            ("throughput/samples_per_second", "Samples/sec"),
        ])

        print(f"{'='*70}\n")

    def _print_category(self, category_name: str, metrics: Dict[str, Any], keys: List[tuple]):
        """打印一个类别的指标"""
        values = []
        for key, display_name in keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    values.append(f"  {display_name}: {value:.6f}" if isinstance(value, float) else f"  {display_name}: {value}")
                else:
                    values.append(f"  {display_name}: {value}")

        if values:
            print(f"\n[{category_name}]")
            for v in values:
                print(v)

    def _save_to_json(self):
        """保存训练历史到JSON文件"""
        try:
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "experiment_name": self.experiment_name,
                    "total_steps": len(self.history),
                    "history": self.history
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[TrainingStateLogger] Error saving to JSON: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.history:
            return {}

        latest = self.history[-1]
        return {
            "current_step": latest["step"],
            "total_records": len(self.history),
            "json_file": self.json_file_path,
        }


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.

    V8 更新: 使用vllm加载模型推理获取next_token_embedding作为数据表征
    """

    @staticmethod
    def _category_from_source(data_source: str) -> str:
        if data_source.startswith("math"):
            return "math"
        if data_source.startswith("code"):
            return "code"
        if data_source.startswith("wildchat"):
            return "general"
        raise NotImplementedError(f"Illegal data source: {data_source}")

    def _dynamic_state_checkpoint_path(self, global_step: int) -> str:
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
        return os.path.join(checkpoint_root, f"global_step_{global_step}", "dynamic_state.pt")

    def _explicit_save_steps(self) -> set:
        save_steps = self.config.trainer.get("save_steps", None)
        if save_steps in (None, "", "null", "None"):
            return set()
        if isinstance(save_steps, str):
            raw = save_steps.strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1]
            items = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            items = list(save_steps)

        steps = set()
        for item in items:
            try:
                steps.add(int(item))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid trainer.save_steps entry: {item!r}") from None
        return steps

    def _should_save_checkpoint_now(self, is_last_step: bool) -> bool:
        explicit_save_hit = self.global_steps in self._explicit_save_steps()
        save_freq = int(self.config.trainer.save_freq)
        periodic_save_hit = save_freq > 0 and (
            is_last_step or self.global_steps % save_freq == 0
        )
        return explicit_save_hit or periodic_save_hit

    def _save_checkpoint(self):
        super()._save_checkpoint()

        if (
            getattr(self, "_dynamic_runtime_state", None) is None
            or getattr(self, "_global_category_pool_sampler", None) is None
        ):
            return

        dynamic_state = {
            "current_weights": self._dynamic_runtime_state["current_weights"].tolist(),
            "category_rewards": self._dynamic_runtime_state["category_rewards"],
            "sampler_state": self._global_category_pool_sampler.state_dict(),
        }
        dynamic_state_path = self._dynamic_state_checkpoint_path(self.global_steps)
        torch.save(dynamic_state, dynamic_state_path)
        print(f"[Dynamic Sampling V8] Saved sampler state to {dynamic_state_path}")

    def _maybe_load_dynamic_state(
        self,
        checkpoint_step: int,
        current_weights: np.ndarray,
        category_rewards: Dict[str, List[float]],
    ) -> tuple[np.ndarray, Dict[str, List[float]]]:
        dynamic_state_path = self._dynamic_state_checkpoint_path(checkpoint_step)
        if not os.path.exists(dynamic_state_path):
            print(f"[Dynamic Sampling V8] No sampler state found at {dynamic_state_path}, starting fresh.")
            return current_weights, category_rewards

        dynamic_state = torch.load(dynamic_state_path, weights_only=False)
        sampler_state = dynamic_state.get("sampler_state")
        if sampler_state is not None:
            self._global_category_pool_sampler.load_state_dict(sampler_state)

        loaded_weights = dynamic_state.get("current_weights")
        if loaded_weights is not None:
            current_weights = np.asarray(loaded_weights, dtype=np.float32)

        loaded_rewards = dynamic_state.get("category_rewards")
        if loaded_rewards is not None:
            category_rewards = {
                cat: list(loaded_rewards.get(cat, [])) for cat in category_rewards
            }

        print(f"[Dynamic Sampling V8] Loaded sampler state from {dynamic_state_path}")
        return current_weights, category_rewards

    def _build_dynamic_batch(self, current_weights: np.ndarray):
        categories = ["math", "code", "general"]
        if getattr(self, "_global_category_pool_sampler", None) is None:
            raise RuntimeError("GlobalCategoryPoolSampler is not initialized.")

        sample_result = self._global_category_pool_sampler.sample_batch(current_weights)
        batch_items = [self.train_dataset[idx] for idx in sample_result["indices"]]

        collate_fn = getattr(self.train_dataloader, "collate_fn", None)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        batch_dict = collate_fn(batch_items)
        batch = DataProto.from_single_dict(batch_dict)

        dynamic_metrics = {}
        for cat in categories:
            dynamic_metrics[f"dynamic/applied_ratio_{cat}"] = float(
                sample_result["ratios"].get(cat, 0.0)
            )
            dynamic_metrics[f"dynamic/pool_remaining_{cat}"] = float(
                sample_result["remaining"].get(cat, 0)
            )
            dynamic_metrics[f"dynamic/pool_remaining_ratio_{cat}"] = float(
                sample_result["remaining_ratio"].get(cat, 0.0)
            )
            dynamic_metrics[f"dynamic/pool_reset_{cat}"] = float(
                sample_result["resets"].get(cat, 0)
            )

        return batch, dynamic_metrics

    def _create_global_category_pool_sampler(self, batch_size: int, seed: int):
        return GlobalCategoryPoolSampler(
            dataset=self.train_dataset,
            batch_size=batch_size,
            seed=seed,
        )

    def _update_dynamic_sample_observables(
        self,
        *,
        new_batch: DataProto,
        uids: np.ndarray,
        unique_uids: np.ndarray,
        fn_reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Dict[str, List[Any]],
        metrics: Dict[str, float],
    ) -> None:
        return None

    @staticmethod
    def _extract_target_text_from_record(record: Any) -> Optional[str]:
        """
        只从题面字段提取目标文本，明确排除 ground truth / answer / output。
        """
        if not isinstance(record, dict):
            return None

        prompt_parts: List[str] = []
        for key in ("instruction", "prompt", "question", "content", "input"):
            value = record.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    prompt_parts.append(text)

        if not prompt_parts:
            return None
        return "\n".join(prompt_parts)

    @classmethod
    def _extract_text_value_from_json_line(cls, line: str) -> Optional[str]:
        """
        从单行 JSON / JSON 片段中提取题面字段。
        """
        stripped = line.strip().rstrip(",")
        if not stripped or stripped in {"[", "]", "{", "}"}:
            return None

        try:
            parsed = json.loads(stripped)
        except Exception:
            wrapped = "{" + stripped + "}"
            try:
                parsed = json.loads(wrapped)
            except Exception:
                return None

        return cls._extract_target_text_from_record(parsed)

    def _load_target_texts_from_file(
        self,
        file_path: str,
        max_lines: int,
    ) -> List[str]:
        """
        读取目标文件并提取题面文本。

        优先解析 JSON / JSON array / JSONL，且只读取 instruction/prompt/question/content/input，
        不使用 output/ground_truth/answer/solution 等答案字段。
        """
        if not file_path or not os.path.exists(file_path):
            print(f"[Dynamic Sampling V8] Target file not found: {file_path}")
            return []

        texts: List[str] = []
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None

        if isinstance(parsed, list):
            for record in parsed[:max_lines]:
                text = self._extract_target_text_from_record(record)
                if text:
                    texts.append(text)
            return texts

        if isinstance(parsed, dict):
            candidate_records = None
            for key in ("data", "examples", "items", "records"):
                value = parsed.get(key)
                if isinstance(value, list):
                    candidate_records = value
                    break

            if candidate_records is None:
                text = self._extract_target_text_from_record(parsed)
                return [text] if text else []

            for record in candidate_records[:max_lines]:
                text = self._extract_target_text_from_record(record)
                if text:
                    texts.append(text)
            return texts

        for line_idx, line in enumerate(raw_text.splitlines()):
            if line_idx >= max_lines:
                break
            text = self._extract_text_value_from_json_line(line)
            if text:
                texts.append(text)
        return texts

    def _build_target_representations_with_vllm(
        self,
        target_test_files: Dict[str, str],
        max_lines: int,
        max_tokens: int,
        model_path: str,
    ) -> Dict[str, np.ndarray]:
        """
        使用vllm加载模型推理获取next_token_embedding作为目标数据表征。

        对于每个目标领域:
        1. 读取max_lines条数据
        2. 使用vllm推理每条数据，获取next_token的hidden state作为embedding
        3. 对所有embedding求平均，作为该领域的表征

        Args:
            target_test_files: 三个目标测试文件路径 {math: path, code: path, general: path}
            max_lines: 每个文件最多读取行数
            max_tokens: tokenizer编码最大token数
            model_path: 模型路径

        Returns:
            Dict[str, np.ndarray]: 三个领域的表征向量 {math: emb, code: emb, general: emb}
        """
        from transformers import AutoTokenizer
        import torch.nn.functional as F

        print(f"[Dynamic Sampling V8] Building target representations with vllm...")
        print(f"  Model path: {model_path}")
        print(f"  Max lines per file: {max_lines}")
        print(f"  Max tokens: {max_tokens}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        target_representations: Dict[str, np.ndarray] = {}

        for category in ("math", "code", "general"):
            file_path = target_test_files.get(category, "")
            texts = self._load_target_texts_from_file(file_path, max_lines=max_lines)

            if not texts:
                print(f"[Dynamic Sampling V8] No texts found for {category}, using zero representation")
                # 获取模型hidden size作为默认维度
                target_representations[category] = np.zeros(4096, dtype=np.float32)  # 默认维度
                continue

            embeddings: List[np.ndarray] = []

            print(f"[Dynamic Sampling V8] Processing {category}: {len(texts)} texts")

            # 使用actor_rollout_wg进行推理获取embedding
            # 这里我们通过调用模型的forward来获取hidden states
            for text in tqdm(texts, desc=f"Encoding {category}"):
                try:
                    # Tokenize
                    encoded = tokenizer(
                        text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_tokens,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"]

                    # 通过actor模型获取hidden states
                    # 使用actor_rollout_wg的模型来获取embedding
                    if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
                        # 构建一个简单的DataProto用于推理
                        from verl import DataProto
                        batch_dict = {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
                        }
                        mini_batch = DataProto.from_single_dict(batch_dict)

                        # 调用模型获取hidden states
                        # 注意：这需要模型支持返回hidden states
                        with torch.no_grad():
                            # 获取最后一个token的hidden state作为表征
                            # 这里使用actor_rollout_wg的compute_hidden_states方法（需要实现）
                            hidden_states = self._get_next_token_embedding(mini_batch)
                            if hidden_states is not None:
                                emb = hidden_states.cpu().numpy().flatten()
                                embeddings.append(emb)
                except Exception as e:
                    print(f"[Dynamic Sampling V8] Error processing text: {e}")
                    continue

            if embeddings:
                # 对所有embedding求平均
                target_rep = np.mean(embeddings, axis=0)
                # L2归一化
                norm = np.linalg.norm(target_rep)
                if norm > 0:
                    target_rep = target_rep / norm
                target_representations[category] = target_rep.astype(np.float32)
                print(f"[Dynamic Sampling V8] {category} representation shape: {target_rep.shape}")
            else:
                print(f"[Dynamic Sampling V8] No valid embeddings for {category}, using zero representation")
                target_representations[category] = np.zeros(4096, dtype=np.float32)

        return target_representations

    def _get_next_token_embedding(self, batch: DataProto) -> Optional[torch.Tensor]:
        """
        获取next token的embedding作为数据表征。

        使用模型的forward pass获取最后一层的hidden states，
        取最后一个token位置的hidden state作为该数据的表征。
        """
        try:
            # 通过actor模型获取hidden states
            # 这里需要调用底层模型的forward方法
            if hasattr(self.actor_rollout_wg, 'actor_module'):
                actor_module = self.actor_rollout_wg.actor_module
                if hasattr(actor_module, 'forward'):
                    with torch.no_grad():
                        input_ids = batch.batch["input_ids"]
                        attention_mask = batch.batch["attention_mask"]

                        # 调用模型forward，获取hidden states
                        outputs = actor_module.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )

                        # 获取最后一层hidden states
                        last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

                        # 取最后一个有效token的hidden state
                        seq_len = attention_mask.sum(dim=-1) - 1  # 最后一个有效token的位置
                        batch_indices = torch.arange(last_hidden_state.shape[0])
                        next_token_emb = last_hidden_state[batch_indices, seq_len]  # [batch, hidden_dim]

                        return next_token_emb.squeeze(0)  # [hidden_dim]
        except Exception as e:
            print(f"[Dynamic Sampling V8] Error getting next token embedding: {e}")
            return None

        return None

    def _compute_batch_category_representations_with_vllm(
        self,
        batch: DataProto,
    ) -> Dict[str, np.ndarray]:
        """
        基于当前batch获取各类别的表征（使用next_token_embedding）。
        """
        data_sources = batch.non_tensor_batch.get("data_source", [])
        if len(data_sources) == 0:
            return {}

        # 获取整个batch的next token embeddings
        next_token_embs = self._get_batch_next_token_embeddings(batch)
        if next_token_embs is None:
            return {}

        cat_to_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
        for idx, data_source in enumerate(data_sources):
            try:
                category = self._category_from_source(data_source)
                if idx < len(next_token_embs):
                    cat_to_embs[category].append(next_token_embs[idx])
            except NotImplementedError:
                continue

        category_representations: Dict[str, np.ndarray] = {}
        for category, embs in cat_to_embs.items():
            if embs:
                category_representations[category] = np.mean(embs, axis=0)
        return category_representations

    def _get_batch_next_token_embeddings(self, batch: DataProto) -> Optional[np.ndarray]:
        """
        获取batch中每个样本的next token embedding。
        """
        try:
            if hasattr(self.actor_rollout_wg, 'actor_module'):
                actor_module = self.actor_rollout_wg.actor_module

                with torch.no_grad():
                    input_ids = batch.batch["input_ids"]
                    attention_mask = batch.batch["attention_mask"]

                    outputs = actor_module.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                    last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

                    # 取每个样本最后一个有效token位置的hidden state
                    seq_lens = attention_mask.sum(dim=-1) - 1
                    batch_indices = torch.arange(last_hidden_state.shape[0])
                    next_token_embs = last_hidden_state[batch_indices, seq_lens]  # [batch, hidden_dim]

                    return next_token_embs.cpu().numpy()
        except Exception as e:
            print(f"[Dynamic Sampling V8] Error getting batch embeddings: {e}")
            return None

        return None

    @staticmethod
    def _compute_distance_sum(
        categories: List[str],
        category_representations: Dict[str, np.ndarray],
        target_representations: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        计算每个训练类别到三个目标表征的距离和:
            d_i = Σ_j ||t_i - s_j||_2
        """
        distances = np.ones(len(categories), dtype=np.float32)
        valid_targets = [
            target_representations.get(cat) for cat in ("math", "code", "general")
            if cat in target_representations and np.any(target_representations[cat])
        ]
        if not valid_targets:
            return distances

        for i, category in enumerate(categories):
            category_rep = category_representations.get(category)
            if category_rep is None or not np.any(category_rep):
                continue
            dist_sum = 0.0
            for target_rep in valid_targets:
                dist_sum += float(np.linalg.norm(category_rep - target_rep))
            distances[i] = dist_sum
        return distances

    def fit(self):
        """
        The training loop of PPO.
        V8: 使用vllm/模型推理获取next_token_embedding作为表征
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
        loaded_global_steps = self.global_steps

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # ========== Dynamic sampling V8 init ==========
        dynamic_config = self.config.trainer.get("dynamic_sampling", {})
        enable_dynamic = dynamic_config.get("enable", False)
        update_freq = dynamic_config.get("update_freq", 20)  # V8: 默认20步更新一次
        min_weight = dynamic_config.get("min_weight", 0.1)
        use_inverse_improvement = dynamic_config.get("use_inverse_improvement", True)

        # V8 参数
        target_read_lines = dynamic_config.get("target_read_lines", 1000)  # V8: 读取1000条
        target_max_tokens = dynamic_config.get("target_max_tokens", 8192)  # V8: 最大8192 tokens
        model_path = self.config.actor_rollout_ref.model.path  # 使用正在训练的模型

        target_files_cfg = dynamic_config.get("target_test_files", {}) or {}
        target_test_files = {
            "math": target_files_cfg.get("math_file", "/root/work/tjshen/ArcherCodeR/data/test/AIME2025.json"),
            "code": target_files_cfg.get("code_file", "/root/work/tjshen/ArcherCodeR/data/test/LCB.json"),
            "general": target_files_cfg.get("general_file", "/root/work/tjshen/ArcherCodeR/data/test/Arena_question.json"),
        }

        # 记录各类别reward历史
        category_rewards = {"math": [], "code": [], "general": []}
        target_representations: Dict[str, np.ndarray] = {}
        current_weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self._global_category_pool_sampler = None
        self._dynamic_runtime_state = None

        if enable_dynamic:
            print("=" * 50)
            print("Dynamic Sampling V8 Enabled")
            print(f"  Using model: {model_path}")
            print(f"  Update frequency: {update_freq} steps")
            print(f"  Min weight: {min_weight}")
            print(f"  Use inverse improvement: {use_inverse_improvement}")
            print(f"  Target read lines: {target_read_lines}")
            print(f"  Target max tokens: {target_max_tokens}")
            print(f"  Target files: {target_test_files}")
            print("  Representation: next_token_embedding from model")
            print("=" * 50)

            # V8: 使用模型推理构建目标表征
            target_representations = self._build_target_representations_with_vllm(
                target_test_files=target_test_files,
                max_lines=target_read_lines,
                max_tokens=target_max_tokens,
                model_path=model_path,
            )
            dynamic_batch_size = self.config.data.get(
                "gen_batch_size", self.config.data.train_batch_size
            )
            dynamic_seed = int(self.config.data.get("seed", 1))
            self._global_category_pool_sampler = self._create_global_category_pool_sampler(
                batch_size=dynamic_batch_size,
                seed=dynamic_seed,
            )
            if loaded_global_steps > 0:
                current_weights, category_rewards = self._maybe_load_dynamic_state(
                    checkpoint_step=loaded_global_steps,
                    current_weights=current_weights,
                    category_rewards=category_rewards,
                )
            self._dynamic_runtime_state = {
                "current_weights": current_weights,
                "category_rewards": category_rewards,
            }
            print("=" * 50)
        # ========== Dynamic sampling V8 init end ==========

        # ========== V8: 初始化训练状态日志记录器 ==========
        output_dir = self.config.trainer.get("default_local_dir", "./output")
        experiment_name = self.config.trainer.get("experiment_name", "dynamic_v8")
        state_logger = TrainingStateLogger(output_dir=output_dir, experiment_name=experiment_name)
        print(f"[TrainingStateLogger] Initialized. JSON file: {state_logger.json_file_path}")
        # ========== V8: 日志记录器初始化结束 ==========

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                if enable_dynamic:
                    new_batch, dynamic_metrics = self._build_dynamic_batch(current_weights)
                    metrics.update(dynamic_metrics)
                    metrics["dynamic/weight_math"] = float(current_weights[0])
                    metrics["dynamic/weight_code"] = float(current_weights[1])
                    metrics["dynamic/weight_general"] = float(current_weights[2])
                else:
                    new_batch = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # compute the num of code and math prompts
                total_math = 0
                total_code = 0
                total_general = 0
                for data_source in new_batch.non_tensor_batch['data_source']:
                    category = self._category_from_source(data_source)
                    if category == "math":
                        total_math += 1
                    elif category == "code":
                        total_code += 1
                    elif category == "general":
                        total_general += 1

                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    uid_list = [str(uuid.uuid4()) for _ in range(len(new_batch.batch))]
                    new_batch.non_tensor_batch["uid"] = np.array(uid_list, dtype=object)

                    # buid a map from uid to data_source
                    data_source_list = new_batch.non_tensor_batch['data_source']
                    uid_to_source_map: Dict[str, str] = dict(zip(uid_list, data_source_list))

                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)


                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        if self.use_rm:
                            model_reward = self.rm_wg.compute_rm_score(new_batch)
                            model_reward_tensor = model_reward.batch.get("rm_scores")

                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            fn_reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            fn_reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}
                            reward_result = {}

                        if not self.use_rm:
                            model_reward_tensor = torch.zeros_like(fn_reward_tensor)
                        reward_tensor = model_reward_tensor + fn_reward_tensor

                        sample_scores = reward_tensor.sum(-1).cpu().tolist()

                        if "thinking_tokens_info" in reward_result:
                            thinking_tokens_infos_dict = reward_result["thinking_tokens_info"]
                            for key_info in list(thinking_tokens_infos_dict.keys()):
                                lst = thinking_tokens_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        thinking_tokens_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        thinking_tokens_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in thinking_tokens_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)

                        if "repetition_info" in reward_result:
                            repetition_infos_dict = reward_result["repetition_info"]
                            for key_info in list(repetition_infos_dict.keys()):
                                lst = repetition_infos_dict[key_info]
                                assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                                for value, score in zip(lst, sample_scores):
                                    if score > 0:
                                        repetition_infos_dict['pos_'+key_info].append(value)
                                    else:
                                        repetition_infos_dict['neg_'+key_info].append(value)

                            for key_info, lst in repetition_infos_dict.items():
                                metrics[key_info] = sum(lst) / len(lst)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]


                    # Rejection sampling based on rewards
                    uids = new_batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)
                    valid_mask = torch.ones(len(uids), dtype=torch.bool)
                    solve_none = 0
                    solve_all = 0
                    total_math_score = 0
                    model_math_score = 0
                    rule_math_score = 0
                    total_code_score = 0
                    model_code_score = 0
                    rule_code_score = 0
                    total_general_score = 0
                    model_general_score = 0
                    rule_general_score = 0
                    for uid in unique_uids:
                        uid_mask = uids == uid
                        uid_rewards = reward_tensor[uid_mask].sum(-1)
                        uid_model_rewards = model_reward_tensor[uid_mask].sum(-1)
                        uid_fn_rewards = fn_reward_tensor[uid_mask].sum(-1)

                        if (not uid_to_source_map[uid].startswith('wildchat')) and (uid_fn_rewards == 0).all():
                            valid_mask[uid_mask] = False
                            solve_none += 1
                        elif (not uid_to_source_map[uid].startswith('wildchat')) and (uid_fn_rewards == 1).all():
                            valid_mask[uid_mask] = False
                            solve_all += 1

                        if uid_to_source_map[uid].startswith('math'):
                            total_math_score += uid_rewards.mean()
                            model_math_score += uid_model_rewards.mean()
                            rule_math_score += uid_fn_rewards.mean()
                        elif uid_to_source_map[uid].startswith('code'):
                            total_code_score += uid_rewards.mean()
                            model_code_score += uid_model_rewards.mean()
                            rule_code_score += uid_fn_rewards.mean()
                        elif uid_to_source_map[uid].startswith('wildchat'):
                            total_general_score += uid_rewards.mean()
                            model_general_score += uid_model_rewards.mean()
                            rule_general_score += uid_fn_rewards.mean()
                        else:
                            raise NotImplementedError(f"Illegal data source")

                    if total_math:
                        metrics['batch/average_math_score'] = total_math_score / total_math
                        metrics['batch/average_math_model_score'] = model_math_score / total_math
                        metrics['batch/average_math_rule_score'] = rule_math_score / total_math
                    if total_code:
                        metrics['batch/average_code_score'] = total_code_score / total_code
                        metrics['batch/average_code_model_score'] = model_code_score / total_code
                        metrics['batch/average_code_rule_score'] = rule_code_score / total_code
                    if total_general:
                        metrics['batch/average_general_score'] = total_general_score / total_general
                        metrics['batch/average_general_model_score'] = model_general_score / total_general
                        metrics['batch/average_general_rule_score'] = rule_general_score / total_general
                    metrics['batch/solve_none'] = solve_none
                    metrics['batch/solve_all'] = solve_all
                    metrics['batch/valid'] = len(unique_uids) - solve_all - solve_none

                    # V8: 每步输出训练指标
                    print(f"\n[Step {self.global_steps}] Training Metrics:")
                    print(f"  valid: {len(unique_uids) - solve_all - solve_none}, solve_none: {solve_none}, solve_all: {solve_all}")
                    if total_math:
                        print(f"  math_score: {total_math_score / total_math:.4f} (model: {model_math_score / total_math:.4f}, rule: {rule_math_score / total_math:.4f})")
                    if total_code:
                        print(f"  code_score: {total_code_score / total_code:.4f} (model: {model_code_score / total_code:.4f}, rule: {rule_code_score / total_code:.4f})")
                    if total_general:
                        print(f"  general_score: {total_general_score / total_general:.4f} (model: {model_general_score / total_general:.4f}, rule: {rule_general_score / total_general:.4f})")

                    # ========== V8: 记录reward历史 ==========
                    if enable_dynamic:
                        if total_math:
                            category_rewards['math'].append(total_math_score / total_math)
                        if total_code:
                            category_rewards['code'].append(total_code_score / total_code)
                        if total_general:
                            category_rewards['general'].append(total_general_score / total_general)
                        self._update_dynamic_sample_observables(
                            new_batch=new_batch,
                            uids=uids,
                            unique_uids=unique_uids,
                            fn_reward_tensor=fn_reward_tensor,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            metrics=metrics,
                        )
                    # ========== V8: 记录结束 ==========

                    if self.config.trainer.rejection_sample:
                        if not valid_mask.any():
                            continue
                        batch = new_batch[valid_mask]

                    max_response_length = batch.batch['responses'].shape[-1]
                    response_mask = batch.batch['attention_mask'][:, -max_response_length:]
                    response_length = response_mask.sum(-1).float()
                    response_clip_mask = ~torch.ge(response_length, max_response_length)
                    metrics['batch/clip_overlong'] = len(batch) - response_clip_mask.sum()
                    if self.config.trainer.enable_overlong_filter:
                        batch = batch[response_clip_mask]

                    def get_sorted_indices(lst):
                        return [index for index, _ in sorted(enumerate(lst), key=lambda x: x[1])]
                    sorted_indices = torch.tensor(get_sorted_indices(batch.non_tensor_batch['index']))
                    batch.reorder(sorted_indices)

                    num_trainer_replicas = self.actor_rollout_wg.world_size
                    if batch.batch['input_ids'].shape[0] < num_trainer_replicas and num_trainer_replicas/batch.batch['input_ids'].shape[0] <= 2:
                        batch = batch.repeat(repeat_times=math.ceil(num_trainer_replicas/batch.batch['input_ids'].shape[0]), interleave=False)

                    max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                    if not max_batch_size:
                        continue
                    batch = batch[:max_batch_size]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )
                        if "inverse_pair" in batch.meta_info:
                            metrics["batch/inverse_pair"] = batch.meta_info["inverse_pair"] / metrics["batch/valid"]

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self._should_save_checkpoint_now(is_last_step):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches

                # ========== Dynamic sampling V8 weight update ==========
                if enable_dynamic and self.global_steps % update_freq == 0:
                    categories = ["math", "code", "general"]
                    reward_improvements = np.ones(3)

                    for i, cat in enumerate(categories):
                        if len(category_rewards[cat]) >= 2:
                            history = category_rewards[cat][-10:]
                            n = len(history)
                            first_half = np.mean(history[: n // 2])
                            second_half = np.mean(history[n // 2 :])
                            improvement = second_half - first_half
                            reward_improvements[i] = max(0.1, min(1.0, improvement + 0.5))

                    # V8: 使用next_token_embedding计算当前batch的类别表征
                    category_representations = self._compute_batch_category_representations_with_vllm(batch)

                    # 目标距离采用三目标表征距离和
                    distances = self._compute_distance_sum(
                        categories=categories,
                        category_representations=category_representations,
                        target_representations=target_representations,
                    )

                    beta = 1.0 / (np.mean(distances) + 1e-8)
                    gamma = 1.0 / (np.mean(reward_improvements) + 1e-8)

                    if use_inverse_improvement:
                        scores = np.exp(-beta * distances - gamma * reward_improvements)
                    else:
                        scores = np.exp(-beta * distances + gamma * reward_improvements)

                    score_sum = float(np.sum(scores))
                    if not np.isfinite(score_sum) or score_sum <= 0:
                        scores = np.ones_like(scores)
                        score_sum = float(np.sum(scores))

                    current_weights = scores / score_sum
                    current_weights = np.maximum(current_weights, min_weight)
                    current_weights = current_weights / np.sum(current_weights)
                    self._dynamic_runtime_state["current_weights"] = current_weights

                    print(f"\n[Dynamic Sampling V8] Step {self.global_steps} Weights:")
                    print(f"   math:    {current_weights[0]:.4f} ({current_weights[0] * 100:.1f}%)")
                    print(f"   code:    {current_weights[1]:.4f} ({current_weights[1] * 100:.1f}%)")
                    print(f"   general: {current_weights[2]:.4f} ({current_weights[2] * 100:.1f}%)")
                    print(
                        f"   dists(sum): math={distances[0]:.4f}, "
                        f"code={distances[1]:.4f}, general={distances[2]:.4f}"
                    )
                    print(f"   beta={beta:.4f}, gamma={gamma:.4f}")

                    metrics["dynamic/weight_math"] = float(current_weights[0])
                    metrics["dynamic/weight_code"] = float(current_weights[1])
                    metrics["dynamic/weight_general"] = float(current_weights[2])
                    metrics["dynamic/improvement_math"] = float(reward_improvements[0])
                    metrics["dynamic/improvement_code"] = float(reward_improvements[1])
                    metrics["dynamic/improvement_general"] = float(reward_improvements[2])
                    metrics["dynamic/distance_math"] = float(distances[0])
                    metrics["dynamic/distance_code"] = float(distances[1])
                    metrics["dynamic/distance_general"] = float(distances[2])
                    metrics["dynamic/beta"] = float(beta)
                    metrics["dynamic/gamma"] = float(gamma)
                # ========== Dynamic sampling V8 update end ==========

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # ========== V8: 记录详细训练状态到JSON ==========
                state_logger.log_step(step=self.global_steps, metrics=metrics, print_detail=True)
                # ========== V8: 记录结束 ==========

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    # 打印训练摘要
                    summary = state_logger.get_summary()
                    print(f"\n[TrainingStateLogger] Training Summary:")
                    print(f"  Total steps recorded: {summary.get('total_records', 0)}")
                    print(f"  JSON file: {summary.get('json_file', 'N/A')}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
