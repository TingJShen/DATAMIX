#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from verl import DataProto

from .dapo_ray_trainer_v8 import RayDAPOTrainer as RayDAPOTrainerV8


class RayDAPOTrainerV11(RayDAPOTrainerV8):
    """
    V11 dynamic sampler.

    Compared with V8, the weight-update stage uses:
    - the full training set to build current category representations
    - the full target/test set to build target category representations

    Sampling itself still reuses the existing global category pool sampler
    so the change is isolated to representation refresh during weight updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v11_tokenizer: Optional[AutoTokenizer] = None
        self._v11_train_texts_by_category: Optional[Dict[str, List[str]]] = None
        self._v11_target_texts_by_category: Optional[Dict[str, List[str]]] = None

    def _v11_hidden_size(self) -> int:
        try:
            if hasattr(self.actor_rollout_wg, "actor_module"):
                actor_module = self.actor_rollout_wg.actor_module
                if hasattr(actor_module, "model") and hasattr(actor_module.model, "config"):
                    return int(getattr(actor_module.model.config, "hidden_size", 4096))
        except Exception:
            pass
        return 4096

    def _v11_config_value(self, key: str, default: Any) -> Any:
        dynamic_config = self.config.trainer.get("dynamic_sampling", {})
        return dynamic_config.get(key, default)

    def _v11_effective_limit(self, key: str, fallback: int = -1) -> int:
        value = int(self._v11_config_value(key, fallback))
        return value

    def _v11_embedding_batch_size(self) -> int:
        value = int(self._v11_config_value("full_dataset_embedding_batch_size", 8))
        return max(1, value)

    def _v11_train_max_tokens(self) -> int:
        return int(
            self._v11_config_value(
                "full_train_max_tokens",
                self.config.data.get("max_prompt_length", 2048),
            )
        )

    def _v11_target_max_tokens(self) -> int:
        return int(
            self._v11_config_value(
                "full_target_max_tokens",
                self._v11_config_value("target_max_tokens", 8192),
            )
        )

    def _v11_get_tokenizer(self, model_path: str) -> AutoTokenizer:
        if self._v11_tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            self._v11_tokenizer = tokenizer
        return self._v11_tokenizer

    @staticmethod
    def _v11_extract_text_chunks(value: Any) -> List[str]:
        chunks: List[str] = []
        if isinstance(value, str):
            text = value.strip()
            if text:
                chunks.append(text)
            return chunks

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        chunks.append(text)
                    continue

                if not isinstance(item, dict):
                    continue

                content = item.get("content")
                if isinstance(content, str):
                    text = content.strip()
                    if text:
                        chunks.append(text)
                    continue

                if isinstance(content, list):
                    for segment in content:
                        if not isinstance(segment, dict):
                            continue
                        if segment.get("type") != "text":
                            continue
                        text = str(segment.get("text", "")).strip()
                        if text:
                            chunks.append(text)
        return chunks

    def _v11_extract_text_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        parts: List[str] = []
        for key in ("prompt", "instruction", "question", "content", "input"):
            parts.extend(self._v11_extract_text_chunks(record.get(key)))

        if not parts:
            return None
        return "\n".join(parts)

    def _v11_collect_training_texts_by_category(self) -> Dict[str, List[str]]:
        if self._v11_train_texts_by_category is not None:
            return self._v11_train_texts_by_category

        limit_per_category = self._v11_effective_limit("full_train_max_samples_per_category", -1)
        collected = {"math": [], "code": [], "general": []}

        dataframe = self.train_dataset.dataframe
        for idx in range(len(dataframe)):
            row = dataframe.iloc[idx].to_dict()
            data_source = str(row.get("data_source", ""))
            try:
                category = self._category_from_source(data_source)
            except NotImplementedError:
                continue

            if limit_per_category > 0 and len(collected[category]) >= limit_per_category:
                continue

            text = self._v11_extract_text_from_record(row)
            if text:
                collected[category].append(text)

        self._v11_train_texts_by_category = collected
        return collected

    def _v11_collect_target_texts_by_category(self, target_test_files: Dict[str, str]) -> Dict[str, List[str]]:
        if self._v11_target_texts_by_category is not None:
            return self._v11_target_texts_by_category

        limit_per_category = self._v11_effective_limit("full_target_max_samples_per_category", -1)
        effective_limit = limit_per_category if limit_per_category > 0 else math.inf

        collected: Dict[str, List[str]] = {}
        for category in ("math", "code", "general"):
            file_path = target_test_files.get(category, "")
            texts = self._load_target_texts_from_file(
                file_path=file_path,
                max_lines=int(effective_limit if effective_limit != math.inf else 10**12),
            )
            collected[category] = texts

        self._v11_target_texts_by_category = collected
        return collected

    def _v11_encode_texts(
        self,
        texts: List[str],
        *,
        tokenizer: AutoTokenizer,
        max_tokens: int,
        batch_size: int,
        desc: str,
    ) -> Optional[np.ndarray]:
        if not texts:
            return None

        all_embeddings: List[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[start : start + batch_size]
            try:
                encoded = tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_tokens,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception as exc:
                print(f"[Dynamic Sampling V11] Tokenization error in {desc}: {exc}")
                continue

            attention_mask = encoded["attention_mask"]
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            mini_batch = DataProto.from_single_dict(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                auto_padding=True,
            )
            batch_embeddings = self._get_batch_next_token_embeddings(mini_batch)
            if batch_embeddings is None:
                continue
            batch_embeddings = batch_embeddings[: len(batch_texts)]
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return None
        return np.concatenate(all_embeddings, axis=0)

    def _get_batch_next_token_embeddings(self, batch: DataProto) -> Optional[np.ndarray]:
        """
        Compute next-token embeddings on the actor workers.

        The driver-side trainer owns a WorkerGroup, not the local actor module.
        V8's direct actor_module access therefore returns None under Ray. V11
        delegates this forward pass to FSDP workers so the representation uses
        the current training model state.
        """
        if hasattr(self.actor_rollout_wg, "compute_next_token_embeddings"):
            try:
                output = self.actor_rollout_wg.compute_next_token_embeddings(batch)
                embeddings = output.batch["next_token_embeddings"]
                return embeddings.detach().cpu().float().numpy()
            except Exception as exc:
                print(f"[Dynamic Sampling V11] Error getting worker embeddings: {exc}")

        return super()._get_batch_next_token_embeddings(batch)

    def _v11_build_representations_from_text_groups(
        self,
        text_groups: Dict[str, List[str]],
        *,
        max_tokens: int,
        label: str,
    ) -> Dict[str, np.ndarray]:
        model_path = self.config.actor_rollout_ref.model.path
        tokenizer = self._v11_get_tokenizer(model_path)
        batch_size = self._v11_embedding_batch_size()
        hidden_size = self._v11_hidden_size()

        representations: Dict[str, np.ndarray] = {}
        for category in ("math", "code", "general"):
            texts = text_groups.get(category, [])
            print(f"[Dynamic Sampling V11] {label} {category}: {len(texts)} texts")

            embeddings = self._v11_encode_texts(
                texts,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                batch_size=batch_size,
                desc=f"{label}-{category}",
            )
            if embeddings is None or len(embeddings) == 0:
                print(f"[Dynamic Sampling V11] No valid embeddings for {label} {category}, using zeros")
                representations[category] = np.zeros(hidden_size, dtype=np.float32)
                continue

            mean_rep = embeddings.mean(axis=0)
            norm = np.linalg.norm(mean_rep)
            if norm > 0:
                mean_rep = mean_rep / norm
            representations[category] = mean_rep.astype(np.float32)
            print(
                f"[Dynamic Sampling V11] {label} {category} representation shape: "
                f"{representations[category].shape}"
            )

        return representations

    def _build_target_representations_with_vllm(
        self,
        target_test_files: Dict[str, str],
        max_lines: int,
        max_tokens: int,
        model_path: str,
    ) -> Dict[str, np.ndarray]:
        # V11 always builds target representations from the full target corpora by default.
        text_groups = self._v11_collect_target_texts_by_category(target_test_files)
        return self._v11_build_representations_from_text_groups(
            text_groups,
            max_tokens=self._v11_target_max_tokens(),
            label="target-full",
        )

    def _compute_batch_category_representations_with_vllm(
        self,
        batch: DataProto,
    ) -> Dict[str, np.ndarray]:
        # V11 ignores the current batch here and refreshes category representations
        # using the full training corpus.
        text_groups = self._v11_collect_training_texts_by_category()
        return self._v11_build_representations_from_text_groups(
            text_groups,
            max_tokens=self._v11_train_max_tokens(),
            label="train-full",
        )

    def _compute_distance_sum(
        self,
        categories: List[str],
        category_representations: Dict[str, np.ndarray],
        target_representations: Dict[str, np.ndarray],
    ) -> np.ndarray:
        # Refresh the full target representations at every weight update so that
        # both sides of the comparison use the latest model state.
        dynamic_config = self.config.trainer.get("dynamic_sampling", {})
        target_files_cfg = dynamic_config.get("target_test_files", {}) or {}
        refreshed_targets = self._build_target_representations_with_vllm(
            target_test_files={
                "math": target_files_cfg.get("math_file", "/root/work/tjshen/ArcherCodeR/data/test/AIME2025.json"),
                "code": target_files_cfg.get("code_file", "/root/work/tjshen/ArcherCodeR/data/test/LCB.json"),
                "general": target_files_cfg.get("general_file", "/root/work/tjshen/ArcherCodeR/data/test/Arena_question.json"),
            },
            max_lines=-1,
            max_tokens=self._v11_target_max_tokens(),
            model_path=self.config.actor_rollout_ref.model.path,
        )
        return RayDAPOTrainerV8._compute_distance_sum(
            categories,
            category_representations,
            refreshed_targets,
        )
