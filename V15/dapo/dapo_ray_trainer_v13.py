#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto

from .dapo_ray_trainer_v11 import RayDAPOTrainerV11
from .dynamic_category_sampler import build_category_indices
from .sample_taylor_sampler import SampleTaylorBatchSampler


class RayDAPOTrainerV13(RayDAPOTrainerV11):
    """
    V13 sample-level Taylor sampler.

    V11's full-dataset target alignment remains the macro budget signal. V13
    adds a sample-level controller over candidate windows using target
    relevance, anchor alignment, learnability, low-rank curvature, and age.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v13_shadow_anchor_indices: Optional[Dict[str, List[int]]] = None
        self._v13_sample_learn_ema: Dict[int, float] = {}
        self._v13_sample_last_seen_step: Dict[int, int] = {}
        self._v13_curvature_matrix_ema: Optional[np.ndarray] = None
        self._v13_anchor_mean_sketch: Optional[np.ndarray] = None
        self._v13_projection_matrix: Optional[np.ndarray] = None
        self._v13_target_representations_cache: Optional[Dict[str, np.ndarray]] = None
        self._v13_target_representations_step: int = -1
        self._v13_anchor_refresh_step: int = -1
        self._v13_domain_budget: Optional[np.ndarray] = None

    def _v13_config_value(self, key: str, default: Any) -> Any:
        dynamic_config = self.config.trainer.get("dynamic_sampling", {})
        return dynamic_config.get(key, default)

    def _v13_projection_dim(self) -> int:
        return max(1, int(self._v13_config_value("grad_projection_dim", 256)))

    def _v13_projection_seed(self) -> int:
        return int(self._v13_config_value("grad_projection_seed", 20260506))

    def _v13_candidate_multiplier(self) -> int:
        return max(1, int(self._v13_config_value("candidate_multiplier", 4)))

    def _v13_repr_batch_size(self) -> int:
        value = int(
            self._v13_config_value(
                "sample_repr_batch_size",
                self._v11_embedding_batch_size(),
            )
        )
        return max(1, value)

    def _v13_sample_temperature(self) -> float:
        return max(1e-6, float(self._v13_config_value("sample_softmax_temperature", 0.7)))

    def _v13_domain_temperature(self) -> float:
        return max(1e-6, float(self._v13_config_value("domain_softmax_temperature", 1.0)))

    def _v13_domain_min_weight(self) -> float:
        return max(0.0, float(self._v13_config_value("domain_min_weight", 0.15)))

    def _v13_learn_ema_decay(self) -> float:
        return float(self._v13_config_value("learn_ema_decay", 0.2))

    def _v13_curvature_refresh_freq(self) -> int:
        update_freq = int(self._v11_config_value("update_freq", 10))
        return max(1, int(self._v13_config_value("curvature_refresh_freq", 5 * update_freq)))

    def _v13_score_weights(self) -> Dict[str, float]:
        default = {
            "target_rel": 1.0,
            "align": 1.0,
            "learn": 0.5,
            "curv": 0.5,
            "age": 0.05,
        }
        cfg = self._v13_config_value("sample_score_weights", {}) or {}
        for key in default:
            if key in cfg:
                default[key] = float(cfg[key])
        return default

    def _create_global_category_pool_sampler(self, batch_size: int, seed: int):
        raw_category_indices = {
            cat: np.asarray(indices, dtype=np.int64)
            for cat, indices in build_category_indices(self.train_dataset).items()
            if cat in ("math", "code", "general")
        }
        self._v13_init_shadow_anchor_indices(raw_category_indices, seed=seed)
        exclude_indices = [
            int(idx)
            for indices in (self._v13_shadow_anchor_indices or {}).values()
            for idx in indices
        ]
        sampler = SampleTaylorBatchSampler(
            dataset=self.train_dataset,
            batch_size=batch_size,
            seed=seed,
            candidate_multiplier=self._v13_candidate_multiplier(),
            sample_softmax_temperature=self._v13_sample_temperature(),
            domain_min_weight=self._v13_domain_min_weight(),
            exclude_indices=exclude_indices,
        )
        return sampler

    def _v13_init_shadow_anchor_indices(self, category_indices: Dict[str, np.ndarray], seed: int):
        if self._v13_shadow_anchor_indices is not None:
            return

        anchor_size = max(1, int(self._v13_config_value("shadow_anchor_size_per_domain", 128)))
        rng = np.random.default_rng(int(seed) + 1009)
        anchors: Dict[str, List[int]] = {}
        for category, indices in category_indices.items():
            if len(indices) == 0:
                anchors[category] = []
                continue
            # Leave at least one sample in the training pool for tiny smoke datasets.
            count = min(anchor_size, max(0, len(indices) - 1))
            if count <= 0:
                anchors[category] = []
                continue
            anchors[category] = rng.choice(indices, size=count, replace=False).astype(np.int64).tolist()
        self._v13_shadow_anchor_indices = anchors
        print(f"[V13] Fixed shadow anchors: { {k: len(v) for k, v in anchors.items()} }")

    def _save_checkpoint(self):
        super()._save_checkpoint()

        dynamic_state_path = self._dynamic_state_checkpoint_path(self.global_steps)
        try:
            dynamic_state = torch.load(dynamic_state_path, weights_only=False)
        except Exception:
            dynamic_state = {}

        dynamic_state.update(
            {
                "v13_shadow_anchor_indices": self._v13_shadow_anchor_indices,
                "v13_sample_learn_ema": {int(k): float(v) for k, v in self._v13_sample_learn_ema.items()},
                "v13_sample_last_seen_step": {
                    int(k): int(v) for k, v in self._v13_sample_last_seen_step.items()
                },
                "v13_projection_seed": self._v13_projection_seed(),
                "v13_curvature_matrix_ema": self._v13_curvature_matrix_ema,
                "v13_anchor_mean_sketch": self._v13_anchor_mean_sketch,
                "v13_domain_budget": self._v13_domain_budget,
            }
        )
        torch.save(dynamic_state, dynamic_state_path)
        print(f"[V13] Saved sample Taylor state to {dynamic_state_path}")

    def _maybe_load_dynamic_state(
        self,
        checkpoint_step: int,
        current_weights: np.ndarray,
        category_rewards: Dict[str, List[float]],
    ) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        current_weights, category_rewards = super()._maybe_load_dynamic_state(
            checkpoint_step,
            current_weights,
            category_rewards,
        )
        dynamic_state_path = self._dynamic_state_checkpoint_path(checkpoint_step)
        try:
            dynamic_state = torch.load(dynamic_state_path, weights_only=False)
        except Exception:
            return current_weights, category_rewards

        self._v13_shadow_anchor_indices = dynamic_state.get("v13_shadow_anchor_indices")
        self._v13_sample_learn_ema = {
            int(k): float(v) for k, v in dynamic_state.get("v13_sample_learn_ema", {}).items()
        }
        self._v13_sample_last_seen_step = {
            int(k): int(v) for k, v in dynamic_state.get("v13_sample_last_seen_step", {}).items()
        }
        self._v13_curvature_matrix_ema = dynamic_state.get("v13_curvature_matrix_ema")
        self._v13_anchor_mean_sketch = dynamic_state.get("v13_anchor_mean_sketch")
        self._v13_domain_budget = dynamic_state.get("v13_domain_budget")
        print(f"[V13] Loaded sample Taylor state from {dynamic_state_path}")
        return current_weights, category_rewards

    def _v13_collate_indices(self, indices: List[int]) -> Optional[DataProto]:
        if not indices:
            return None
        batch_items = [self.train_dataset[int(idx)] for idx in indices]
        collate_fn = getattr(self.train_dataloader, "collate_fn", None)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        return DataProto.from_single_dict(collate_fn(batch_items))

    def _v13_category_for_index(self, dataset_idx: int) -> str:
        row = self.train_dataset.dataframe.iloc[int(dataset_idx)]
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        data_source = str(row.get("data_source", ""))
        return self._category_from_source(data_source)

    def _v13_get_target_representations(self) -> Dict[str, np.ndarray]:
        refresh_freq = max(1, int(self._v13_config_value("target_repr_refresh_freq", self._v13_curvature_refresh_freq())))
        if (
            self._v13_target_representations_cache is not None
            and self._v13_target_representations_step >= 0
            and self.global_steps - self._v13_target_representations_step < refresh_freq
        ):
            return self._v13_target_representations_cache

        dynamic_config = self.config.trainer.get("dynamic_sampling", {})
        target_files_cfg = dynamic_config.get("target_test_files", {}) or {}
        target_representations = self._build_target_representations_with_vllm(
            target_test_files={
                "math": target_files_cfg.get("math_file", "/root/work/tjshen/ArcherCodeR/data/test/AIME2025.json"),
                "code": target_files_cfg.get("code_file", "/root/work/tjshen/ArcherCodeR/data/test/LCB.json"),
                "general": target_files_cfg.get("general_file", "/root/work/tjshen/ArcherCodeR/data/test/Arena_question.json"),
            },
            max_lines=-1,
            max_tokens=self._v11_target_max_tokens(),
            model_path=self.config.actor_rollout_ref.model.path,
        )
        self._v13_target_representations_cache = target_representations
        self._v13_target_representations_step = int(self.global_steps)
        return target_representations

    def _v13_fallback_projection(self, embeddings: np.ndarray) -> np.ndarray:
        projection_dim = self._v13_projection_dim()
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(len(embeddings), -1)
        hidden_size = int(embeddings.shape[-1])
        if self._v13_projection_matrix is None or self._v13_projection_matrix.shape != (hidden_size, projection_dim):
            rng = np.random.default_rng(self._v13_projection_seed())
            self._v13_projection_matrix = (
                rng.normal(size=(hidden_size, projection_dim)).astype(np.float32) / np.sqrt(float(hidden_size))
            )
        sketch = embeddings.astype(np.float32) @ self._v13_projection_matrix
        return sketch.astype(np.float32)

    def _v13_get_repr_and_grad_sketch(
        self,
        indices: List[int],
        target_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not indices:
            return None, None

        repr_batch_size = self._v13_repr_batch_size()
        if len(indices) > repr_batch_size:
            embedding_parts: List[np.ndarray] = []
            sketch_parts: List[np.ndarray] = []
            target_array = None
            if target_embeddings is not None:
                target_array = np.asarray(target_embeddings, dtype=np.float32)

            for start in range(0, len(indices), repr_batch_size):
                end = start + repr_batch_size
                chunk_indices = indices[start:end]
                chunk_targets = None if target_array is None else target_array[start:end]
                chunk_embeddings, chunk_sketch = self._v13_get_repr_and_grad_sketch(
                    chunk_indices,
                    target_embeddings=chunk_targets,
                )
                if chunk_embeddings is None or chunk_sketch is None:
                    return None, None
                embedding_parts.append(chunk_embeddings[: len(chunk_indices)])
                sketch_parts.append(chunk_sketch[: len(chunk_indices)])

            if not embedding_parts or not sketch_parts:
                return None, None
            return (
                np.concatenate(embedding_parts, axis=0).astype(np.float32),
                np.concatenate(sketch_parts, axis=0).astype(np.float32),
            )

        batch = self._v13_collate_indices(indices)
        if batch is None:
            return None, None
        if target_embeddings is not None:
            batch.batch["v13_target_embeddings"] = torch.as_tensor(target_embeddings, dtype=torch.float32)
        if getattr(batch, "meta_info", None) is None:
            batch.meta_info = {}
        batch.meta_info["v13_grad_projection_dim"] = self._v13_projection_dim()
        batch.meta_info["v13_grad_projection_seed"] = self._v13_projection_seed()

        if hasattr(self.actor_rollout_wg, "compute_v13_repr_and_grad_sketch"):
            try:
                wg_size = getattr(self.actor_rollout_wg, 'world_size', 2)
                actual_len = len(batch)
                if actual_len < wg_size:
                    pad_count = wg_size - actual_len
                    pad_batch = batch[:1]
                    for _ in range(pad_count - 1):
                        pad_batch = DataProto.concat([pad_batch, batch[:1]])
                    batch = DataProto.concat([batch, pad_batch])
                elif actual_len % wg_size != 0:
                    pad_count = wg_size - (actual_len % wg_size)
                    pad_batch = batch[:pad_count]
                    batch = DataProto.concat([batch, pad_batch])
                output = self.actor_rollout_wg.compute_v13_repr_and_grad_sketch(batch)
                embeddings = output.batch["next_token_embeddings"].detach().cpu().float().numpy()
                grad_sketch = output.batch["grad_sketch"].detach().cpu().float().numpy()
                return embeddings[: len(indices)], grad_sketch[: len(indices)]
            except Exception as exc:
                print(f"[V13] Worker grad sketch failed, using embedding projection fallback: {exc}")

        embeddings = self._get_batch_next_token_embeddings(batch)
        if embeddings is None:
            return None, None
        embeddings = embeddings[: len(indices)]
        return embeddings, self._v13_fallback_projection(embeddings)

    def _v13_maybe_refresh_anchor_statistics(self, force: bool = False) -> bool:
        if self._v13_shadow_anchor_indices is None:
            sampler = getattr(self, "_global_category_pool_sampler", None)
            if sampler is not None and hasattr(sampler, "category_indices"):
                self._v13_init_shadow_anchor_indices(sampler.category_indices, seed=int(self.config.data.get("seed", 1)))
        if not self._v13_shadow_anchor_indices:
            return False

        refresh_freq = self._v13_curvature_refresh_freq()
        if (
            not force
            and self._v13_anchor_mean_sketch is not None
            and self._v13_anchor_refresh_step >= 0
            and self.global_steps - self._v13_anchor_refresh_step < refresh_freq
        ):
            return False

        anchor_indices = [
            int(idx)
            for indices in self._v13_shadow_anchor_indices.values()
            for idx in indices
        ]
        target_representations = self._v13_get_target_representations()
        target_embeddings = []
        for idx in anchor_indices:
            category = self._v13_category_for_index(int(idx))
            target_embeddings.append(target_representations.get(category, np.zeros(self._v11_hidden_size(), dtype=np.float32)))
        _, grad_sketch = self._v13_get_repr_and_grad_sketch(
            anchor_indices,
            target_embeddings=np.asarray(target_embeddings, dtype=np.float32),
        )
        if grad_sketch is None or len(grad_sketch) == 0:
            return False

        grad_sketch = grad_sketch.astype(np.float32)
        self._v13_anchor_mean_sketch = grad_sketch.mean(axis=0).astype(np.float32)
        curvature = (grad_sketch.T @ grad_sketch) / max(1, len(grad_sketch))
        curvature = 0.5 * (curvature + curvature.T)
        if self._v13_curvature_matrix_ema is None:
            self._v13_curvature_matrix_ema = curvature.astype(np.float32)
        else:
            decay = float(self._v13_config_value("curvature_ema_decay", 0.9))
            self._v13_curvature_matrix_ema = (
                decay * self._v13_curvature_matrix_ema + (1.0 - decay) * curvature
            ).astype(np.float32)
        self._v13_anchor_refresh_step = int(self.global_steps)
        return True

    @staticmethod
    def _v13_normalize_component(values: Dict[int, float], indices: List[int]) -> Dict[int, float]:
        if not indices:
            return {}
        arr = np.asarray([float(values.get(int(idx), 0.0)) for idx in indices], dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            return {int(idx): 0.0 for idx in indices}
        return {int(idx): float((float(values.get(int(idx), 0.0)) - mean) / std) for idx in indices}

    def _v13_domain_learn_default(self, category: str) -> float:
        sampler = getattr(self, "_global_category_pool_sampler", None)
        if sampler is None or not hasattr(sampler, "category_indices"):
            return 0.5
        values = [
            self._v13_sample_learn_ema[int(idx)]
            for idx in sampler.category_indices.get(category, [])
            if int(idx) in self._v13_sample_learn_ema
        ]
        return float(np.mean(values)) if values else 0.5

    def _v13_compute_candidate_scores(
        self,
        candidate_indices_by_category: Dict[str, List[int]],
    ) -> Tuple[Dict[int, float], Dict[str, Dict[str, float]]]:
        self._v13_maybe_refresh_anchor_statistics(force=self._v13_anchor_mean_sketch is None)
        target_representations = self._v13_get_target_representations()
        flat_indices = [
            int(idx)
            for indices in candidate_indices_by_category.values()
            for idx in indices
        ]
        target_embeddings = []
        for idx in flat_indices:
            category = self._v13_category_for_index(int(idx))
            target_embeddings.append(target_representations.get(category, np.zeros(self._v11_hidden_size(), dtype=np.float32)))
        embeddings, grad_sketch = self._v13_get_repr_and_grad_sketch(
            flat_indices,
            target_embeddings=np.asarray(target_embeddings, dtype=np.float32),
        )

        raw_components = {
            "target_rel": {},
            "align": {},
            "learn": {},
            "curv": {},
            "age": {},
        }
        if embeddings is None or grad_sketch is None:
            embeddings = np.zeros((len(flat_indices), self._v11_hidden_size()), dtype=np.float32)
            grad_sketch = np.zeros((len(flat_indices), self._v13_projection_dim()), dtype=np.float32)

        index_to_pos = {int(idx): pos for pos, idx in enumerate(flat_indices)}
        grad_sketch_dict: Dict[int, "np.ndarray"] = {}
        if grad_sketch is not None:
            for idx, pos in index_to_pos.items():
                grad_sketch_dict[idx] = grad_sketch[pos]

        for category, indices in candidate_indices_by_category.items():
            target_rep = target_representations.get(category)
            target_norm = float(np.linalg.norm(target_rep)) if target_rep is not None else 0.0
            learn_default = self._v13_domain_learn_default(category)
            for dataset_idx in indices:
                dataset_idx = int(dataset_idx)
                pos = index_to_pos[dataset_idx]
                emb = embeddings[pos].astype(np.float32)
                emb_norm = float(np.linalg.norm(emb))
                if target_rep is not None and emb_norm > 0.0 and target_norm > 0.0:
                    raw_components["target_rel"][dataset_idx] = float(np.dot(emb / emb_norm, target_rep / target_norm))
                else:
                    raw_components["target_rel"][dataset_idx] = 0.0

                z = grad_sketch[pos].astype(np.float32)
                if self._v13_anchor_mean_sketch is not None:
                    raw_components["align"][dataset_idx] = float(np.dot(z, self._v13_anchor_mean_sketch))
                else:
                    raw_components["align"][dataset_idx] = 0.0
                if self._v13_curvature_matrix_ema is not None:
                    raw_components["curv"][dataset_idx] = float(z @ self._v13_curvature_matrix_ema @ z)
                else:
                    raw_components["curv"][dataset_idx] = 0.0

                raw_components["learn"][dataset_idx] = float(self._v13_sample_learn_ema.get(dataset_idx, learn_default))
                last_seen = self._v13_sample_last_seen_step.get(dataset_idx)
                raw_components["age"][dataset_idx] = float(max(1, self.global_steps + 1 if last_seen is None else self.global_steps - last_seen))

        weights = self._v13_score_weights()
        sample_scores: Dict[int, float] = {}
        summary: Dict[str, Dict[str, float]] = {}

        for category, indices in candidate_indices_by_category.items():
            cat_indices = [int(idx) for idx in indices]
            normalized = {
                name: self._v13_normalize_component(component, cat_indices)
                for name, component in raw_components.items()
            }
            for dataset_idx in cat_indices:
                score = (
                    weights["target_rel"] * normalized["target_rel"].get(dataset_idx, 0.0)
                    + weights["align"] * normalized["align"].get(dataset_idx, 0.0)
                    + weights["learn"] * normalized["learn"].get(dataset_idx, 0.0)
                    - weights["curv"] * normalized["curv"].get(dataset_idx, 0.0)
                    + weights["age"] * normalized["age"].get(dataset_idx, 0.0)
                )
                sample_scores[dataset_idx] = float(score)

            summary[category] = {}
            for name, component in raw_components.items():
                values = [float(component.get(idx, 0.0)) for idx in cat_indices]
                summary[category][name] = float(np.mean(values)) if values else 0.0
            values = [sample_scores[idx] for idx in cat_indices]
            summary[category]["score"] = float(np.mean(values)) if values else 0.0
            if values:
                top_k = max(1, int(np.ceil(0.25 * len(values))))
                summary[category]["score_top_mean"] = float(np.mean(sorted(values, reverse=True)[:top_k]))
            else:
                summary[category]["score_top_mean"] = 0.0
        return sample_scores, summary, grad_sketch_dict

    def _v13_budget_weights(self, current_weights: np.ndarray, score_summary: Dict[str, Dict[str, float]]) -> np.ndarray:
        categories = getattr(self._global_category_pool_sampler, "categories", ["math", "code", "general"])
        external = np.asarray(list(current_weights), dtype=np.float64)[: len(categories)]
        external = np.maximum(external, 0.0)
        if float(external.sum()) <= 0.0:
            external = np.ones(len(categories), dtype=np.float64) / float(len(categories))
        else:
            external = external / float(external.sum())

        top_scores = np.asarray(
            [float(score_summary.get(cat, {}).get("score_top_mean", 0.0)) for cat in categories],
            dtype=np.float64,
        )
        scaled = top_scores / self._v13_domain_temperature()
        scaled = scaled - float(np.max(scaled))
        proposed = np.exp(np.clip(scaled, -60.0, 60.0))
        proposed = proposed / float(proposed.sum()) if float(proposed.sum()) > 0.0 else external

        gamma = float(self._v13_config_value("domain_budget_smooth", 0.2))
        base = external if self._v13_domain_budget is None else np.asarray(self._v13_domain_budget, dtype=np.float64)
        mixed = (1.0 - gamma) * base + gamma * proposed
        floor = min(self._v13_domain_min_weight(), 1.0 / float(len(categories)))
        if floor > 0.0 and floor * len(categories) < 1.0:
            mixed = floor + (1.0 - floor * len(categories)) * (mixed / float(mixed.sum()))
        mixed = mixed / float(mixed.sum())
        self._v13_domain_budget = mixed.astype(np.float32)
        return self._v13_domain_budget

    def _build_dynamic_batch(self, current_weights: np.ndarray):
        if getattr(self, "_global_category_pool_sampler", None) is None:
            raise RuntimeError("SampleTaylorBatchSampler is not initialized.")

        initial_peek = self._global_category_pool_sampler.peek_candidates(current_weights)
        initial_scores, initial_summary, initial_sketch_dict = self._v13_compute_candidate_scores(
            initial_peek["candidate_indices_by_category"]
        )
        v13_weights = self._v13_budget_weights(current_weights, initial_summary)
        final_peek = self._global_category_pool_sampler.peek_candidates(v13_weights)
        if final_peek["candidate_indices_by_category"] == initial_peek["candidate_indices_by_category"]:
            sample_scores, score_summary, grad_sketch_dict = initial_scores, initial_summary, initial_sketch_dict
        else:
            sample_scores, score_summary, grad_sketch_dict = self._v13_compute_candidate_scores(
                final_peek["candidate_indices_by_category"]
            )
        sample_result = self._global_category_pool_sampler.sample_batch(
            domain_weights=v13_weights,
            sample_scores=sample_scores,
            grad_sketches=grad_sketch_dict,
            curvature_matrix=self._v13_curvature_matrix_ema,
        )

        batch_items = [self.train_dataset[idx] for idx in sample_result["indices"]]
        collate_fn = getattr(self.train_dataloader, "collate_fn", None)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        batch_dict = collate_fn(batch_items)
        batch_dict["v13_dataset_index"] = np.asarray(sample_result["indices"], dtype=np.int64)
        batch_dict["v13_sample_score"] = np.asarray(
            [float(sample_scores.get(int(idx), 0.0)) for idx in sample_result["indices"]],
            dtype=np.float32,
        )
        batch = DataProto.from_single_dict(batch_dict)

        dynamic_metrics: Dict[str, float] = {}
        for category in self._global_category_pool_sampler.categories:
            dynamic_metrics[f"dynamic/v13_weight_{category}"] = float(
                sample_result["normalized_weights"].get(category, 0.0)
            )
            dynamic_metrics[f"dynamic/v13_candidate_count_{category}"] = float(
                sample_result["candidate_counts"].get(category, 0)
            )
            dynamic_metrics[f"dynamic/v13_score_mean_{category}"] = float(
                score_summary.get(category, {}).get("score", 0.0)
            )
            for component in ("target_rel", "align", "learn", "curv", "age"):
                dynamic_metrics[f"dynamic/v13_{component}_mean_{category}"] = float(
                    score_summary.get(category, {}).get(component, 0.0)
                )
            dynamic_metrics[f"dynamic/v13_pool_remaining_{category}"] = float(
                sample_result["remaining"].get(category, 0)
            )
            dynamic_metrics[f"dynamic/v13_pool_reset_{category}"] = float(
                sample_result["resets"].get(category, 0)
            )
        dynamic_metrics["dynamic/v13_anchor_refresh"] = 1.0 if self._v13_anchor_refresh_step == self.global_steps else 0.0
        dynamic_metrics["dynamic/v13_curvature_refresh"] = dynamic_metrics["dynamic/v13_anchor_refresh"]
        if grad_sketch_dict and self._v13_curvature_matrix_ema is not None:
            selected_indices = sample_result["indices"]
            if len(selected_indices) > 1:
                sel_sketches = [grad_sketch_dict[int(idx)] for idx in selected_indices if int(idx) in grad_sketch_dict]
                if len(sel_sketches) > 1:
                    sel_arr = np.stack(sel_sketches, axis=0)
                    C = self._v13_curvature_matrix_ema
                    cross = float(np.mean([float(sel_arr[i] @ C @ sel_arr[j])
                                           for i in range(len(sel_arr)) for j in range(i+1, len(sel_arr))]))
                    dynamic_metrics["dynamic/v15_batch_cross_interaction"] = cross
        print(f"[V13] step={self.global_steps} weights={sample_result['normalized_weights']}")
        return batch, dynamic_metrics

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
        if "v13_dataset_index" not in new_batch.non_tensor_batch:
            return

        dataset_indices = np.asarray(new_batch.non_tensor_batch["v13_dataset_index"], dtype=np.int64)
        ema_decay = self._v13_learn_ema_decay()
        updated = 0
        for uid in unique_uids:
            uid_mask = uids == uid
            prompt_indices = dataset_indices[uid_mask]
            if len(prompt_indices) == 0:
                continue
            dataset_idx = int(prompt_indices[0])
            avg_rule_reward = float(fn_reward_tensor[uid_mask].sum(-1).mean().item())
            old = self._v13_sample_learn_ema.get(dataset_idx, avg_rule_reward)
            self._v13_sample_learn_ema[dataset_idx] = (
                (1.0 - ema_decay) * float(old) + ema_decay * avg_rule_reward
            )
            self._v13_sample_last_seen_step[dataset_idx] = int(self.global_steps)
            updated += 1

        if updated:
            metrics["dynamic/v13_sample_state_updates"] = float(updated)
