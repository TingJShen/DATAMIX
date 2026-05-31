#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .dynamic_category_sampler import build_category_indices


class SampleTaylorBatchSampler:
    """
    V13 sample-level sampler.

    Domain weights only allocate a per-domain budget. Inside each domain, a
    candidate window is scored at the sample level and sampled without
    replacement. Unselected candidates stay available for later windows.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 1,
        categories: Optional[List[str]] = None,
        candidate_multiplier: int = 4,
        sample_softmax_temperature: float = 0.7,
        domain_min_weight: float = 0.15,
        exclude_indices: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.sample_softmax_temperature = max(1e-6, float(sample_softmax_temperature))
        self.domain_min_weight = max(0.0, float(domain_min_weight))

        raw_category_indices = build_category_indices(dataset)
        exclude_set = {int(idx) for idx in (exclude_indices or [])}
        preferred_categories = categories or ["math", "code", "general"]
        self.categories = []
        self.category_indices = {}
        for cat in preferred_categories:
            kept = [int(idx) for idx in raw_category_indices.get(cat, []) if int(idx) not in exclude_set]
            if kept:
                self.categories.append(cat)
                self.category_indices[cat] = np.asarray(kept, dtype=np.int64)
        self.exclude_indices = sorted(exclude_set)

        self.remaining_orders: Dict[str, List[int]] = {}
        self.reset_counts = {cat: 0 for cat in self.categories}
        self.total_drawn = {cat: 0 for cat in self.categories}
        for cat in self.categories:
            self._reset_category(cat, initial=True)

        samples_per_category = {cat: int(len(indices)) for cat, indices in self.category_indices.items()}
        print("SampleTaylorBatchSampler initialized:")
        print(f"  Categories: {self.categories}")
        print(f"  Samples per category: {samples_per_category}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Candidate multiplier: {self.candidate_multiplier}")

    def _reset_category(self, category: str, initial: bool = False):
        indices = self.category_indices[category]
        self.remaining_orders[category] = self.rng.permutation(indices).astype(np.int64).tolist()
        if not initial:
            self.reset_counts[category] += 1

    def _normalize_domain_weights(self, domain_weights: Any) -> Dict[str, float]:
        if isinstance(domain_weights, dict):
            raw = np.asarray([float(domain_weights.get(cat, 0.0)) for cat in self.categories], dtype=np.float64)
        else:
            values = list(domain_weights)
            if len(values) < len(self.categories):
                raise ValueError(f"Expected at least {len(self.categories)} weights, got {len(values)}")
            raw = np.asarray([float(values[i]) for i in range(len(self.categories))], dtype=np.float64)

        raw = np.maximum(raw, 0.0)
        if float(raw.sum()) <= 0.0:
            normalized = np.ones(len(self.categories), dtype=np.float64) / float(len(self.categories))
        else:
            normalized = raw / float(raw.sum())

        floor = min(self.domain_min_weight, 1.0 / float(len(self.categories)))
        if floor <= 0.0:
            final = normalized
        elif floor * len(self.categories) >= 1.0:
            final = np.ones(len(self.categories), dtype=np.float64) / float(len(self.categories))
        else:
            final = floor + (1.0 - floor * len(self.categories)) * normalized

        final = final / float(final.sum())
        return {cat: float(final[i]) for i, cat in enumerate(self.categories)}

    def _allocate_target_counts(self, normalized_weights: Dict[str, float]) -> Dict[str, int]:
        target_counts = {cat: 0 for cat in self.categories}
        residuals = []
        allocated = 0

        for cat in self.categories:
            raw_target = normalized_weights[cat] * self.batch_size
            int_target = int(np.floor(raw_target))
            target_counts[cat] = int_target
            allocated += int_target
            residuals.append((raw_target - int_target, cat))

        remaining = self.batch_size - allocated
        residuals.sort(key=lambda item: item[0], reverse=True)
        for i in range(remaining):
            target_counts[residuals[i % len(residuals)][1]] += 1
        return target_counts

    def _ensure_target_available(self, category: str, target_num: int):
        if target_num <= 0:
            return
        if len(self.category_indices[category]) == 0:
            raise ValueError(f"Category {category} has no available samples.")
        if len(self.remaining_orders[category]) < min(target_num, len(self.category_indices[category])):
            self._reset_category(category)

    def _softmax_select_without_replacement(
        self,
        candidates: List[int],
        sample_scores: Dict[int, float],
        target_num: int,
    ) -> List[int]:
        if target_num <= 0:
            return []
        if target_num >= len(candidates):
            return list(candidates)

        remaining = list(candidates)
        selected: List[int] = []
        while len(selected) < target_num:
            scores = np.asarray([float(sample_scores.get(int(idx), 0.0)) for idx in remaining], dtype=np.float64)
            scaled = scores / self.sample_softmax_temperature
            scaled = scaled - float(np.max(scaled))
            probs = np.exp(np.clip(scaled, -60.0, 60.0))
            probs_sum = float(probs.sum())
            if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                probs = np.ones(len(remaining), dtype=np.float64) / float(len(remaining))
            else:
                probs = probs / probs_sum
            chosen_pos = int(self.rng.choice(len(remaining), p=probs))
            selected.append(int(remaining.pop(chosen_pos)))
        return selected

    def _batch_aware_select_without_replacement(
        self,
        candidates: List[int],
        sample_scores: Dict[int, float],
        grad_sketches: Optional[Dict[int, np.ndarray]],
        curvature_matrix: Optional[np.ndarray],
        target_num: int,
    ) -> List[int]:
        """Batch-aware selection: penalizes redundancy with already-selected samples."""
        if grad_sketches is None or curvature_matrix is None:
            return self._softmax_select_without_replacement(candidates, sample_scores, target_num)
        if target_num <= 0:
            return []
        if target_num >= len(candidates):
            return list(candidates)

        remaining = list(candidates)
        selected: List[int] = []
        dim = curvature_matrix.shape[0]
        z_Bk = np.zeros(dim, dtype=np.float32)

        while len(selected) < target_num:
            marginal_scores = np.empty(len(remaining), dtype=np.float64)
            for j, idx in enumerate(remaining):
                base_score = float(sample_scores.get(int(idx), 0.0))
                z_j = grad_sketches.get(int(idx))
                if z_j is not None and len(selected) > 0:
                    interaction = float(z_j @ curvature_matrix @ z_Bk)
                    marginal_scores[j] = base_score - interaction
                else:
                    marginal_scores[j] = base_score

            scaled = marginal_scores / self.sample_softmax_temperature
            scaled = scaled - float(np.max(scaled))
            probs = np.exp(np.clip(scaled, -60.0, 60.0))
            probs_sum = float(probs.sum())
            if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                probs = np.ones(len(remaining), dtype=np.float64) / float(len(remaining))
            else:
                probs = probs / probs_sum

            chosen_pos = int(self.rng.choice(len(remaining), p=probs))
            chosen_idx = int(remaining.pop(chosen_pos))
            selected.append(chosen_idx)

            z_chosen = grad_sketches.get(chosen_idx)
            if z_chosen is not None:
                z_Bk = (z_Bk * (len(selected) - 1) + z_chosen.astype(np.float32)) / len(selected)

        return selected

    def _draw_from_category(
        self,
        category: str,
        target_num: int,
        sample_scores: Dict[int, float],
        grad_sketches: Optional[Dict[int, np.ndarray]] = None,
        curvature_matrix: Optional[np.ndarray] = None,
    ) -> tuple[List[int], int, int, float]:
        if target_num <= 0:
            return [], 0, 0, 0.0

        resets_before = self.reset_counts[category]
        self._ensure_target_available(category, target_num)
        reset_events = self.reset_counts[category] - resets_before

        candidate_count = min(
            len(self.remaining_orders[category]),
            max(target_num, self.candidate_multiplier * target_num),
        )
        candidates = list(self.remaining_orders[category][:candidate_count])
        selected = self._batch_aware_select_without_replacement(
            candidates, sample_scores, grad_sketches, curvature_matrix, target_num
        )
        selected_set = set(selected)
        self.remaining_orders[category] = [
            int(idx) for idx in self.remaining_orders[category] if int(idx) not in selected_set
        ]
        self.total_drawn[category] += len(selected)

        score_mean = 0.0
        if candidates:
            score_mean = float(np.mean([float(sample_scores.get(int(idx), 0.0)) for idx in candidates]))
        return selected, reset_events, candidate_count, score_mean

    def peek_candidates(self, domain_weights: Any) -> Dict[str, Any]:
        normalized_weights = self._normalize_domain_weights(domain_weights)
        target_counts = self._allocate_target_counts(normalized_weights)
        candidate_indices_by_category: Dict[str, List[int]] = {}

        for cat in self.categories:
            target_num = target_counts[cat]
            if target_num <= 0:
                candidate_indices_by_category[cat] = []
                continue
            candidate_count = min(
                len(self.remaining_orders[cat]),
                max(target_num, self.candidate_multiplier * target_num),
            )
            candidate_indices_by_category[cat] = list(self.remaining_orders[cat][:candidate_count])

        return {
            "normalized_weights": normalized_weights,
            "target_counts": target_counts,
            "candidate_indices_by_category": candidate_indices_by_category,
        }

    def sample_batch(
        self,
        domain_weights: Any,
        sample_scores: Dict[int, float],
        grad_sketches: Optional[Dict[int, np.ndarray]] = None,
        curvature_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        normalized_weights = self._normalize_domain_weights(domain_weights)
        target_counts = self._allocate_target_counts(normalized_weights)

        sampled_indices: List[int] = []
        sampled_counts = {cat: 0 for cat in self.categories}
        reset_events = {cat: 0 for cat in self.categories}
        candidate_counts = {cat: 0 for cat in self.categories}
        score_means = {cat: 0.0 for cat in self.categories}

        for cat in self.categories:
            chosen, reset_times, candidate_count, score_mean = self._draw_from_category(
                cat,
                target_counts[cat],
                sample_scores=sample_scores,
                grad_sketches=grad_sketches,
                curvature_matrix=curvature_matrix,
            )
            sampled_indices.extend(chosen)
            sampled_counts[cat] = len(chosen)
            reset_events[cat] = reset_times
            candidate_counts[cat] = candidate_count
            score_means[cat] = score_mean

        sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
        self.rng.shuffle(sampled_indices)

        ratios = {
            cat: float(sampled_counts[cat]) / float(self.batch_size) for cat in self.categories
        }
        remaining = {cat: int(len(self.remaining_orders[cat])) for cat in self.categories}
        remaining_ratio = {
            cat: float(remaining[cat]) / float(len(self.category_indices[cat]))
            if len(self.category_indices[cat]) > 0
            else 0.0
            for cat in self.categories
        }

        return {
            "indices": sampled_indices.tolist(),
            "counts": sampled_counts,
            "ratios": ratios,
            "remaining": remaining,
            "remaining_ratio": remaining_ratio,
            "resets": reset_events,
            "total_resets": self.reset_counts.copy(),
            "target_counts": target_counts,
            "candidate_counts": candidate_counts,
            "score_means": score_means,
            "normalized_weights": normalized_weights,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seed": self.seed,
            "categories": list(self.categories),
            "category_indices": {
                cat: indices.tolist() for cat, indices in self.category_indices.items()
            },
            "remaining_orders": {
                cat: list(indices) for cat, indices in self.remaining_orders.items()
            },
            "reset_counts": dict(self.reset_counts),
            "total_drawn": dict(self.total_drawn),
            "rng_state": self.rng.bit_generator.state,
            "candidate_multiplier": self.candidate_multiplier,
            "sample_softmax_temperature": self.sample_softmax_temperature,
            "domain_min_weight": self.domain_min_weight,
            "exclude_indices": list(self.exclude_indices),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        saved_categories = list(state_dict.get("categories", []))
        if saved_categories != self.categories:
            raise ValueError(f"Category mismatch when loading sampler state: {saved_categories} != {self.categories}")

        self.remaining_orders = {
            cat: [int(idx) for idx in state_dict["remaining_orders"][cat]] for cat in self.categories
        }
        self.reset_counts = {
            cat: int(state_dict["reset_counts"][cat]) for cat in self.categories
        }
        self.total_drawn = {
            cat: int(state_dict.get("total_drawn", {}).get(cat, 0)) for cat in self.categories
        }
        self.candidate_multiplier = int(state_dict.get("candidate_multiplier", self.candidate_multiplier))
        self.sample_softmax_temperature = float(
            state_dict.get("sample_softmax_temperature", self.sample_softmax_temperature)
        )
        self.domain_min_weight = float(state_dict.get("domain_min_weight", self.domain_min_weight))
        self.exclude_indices = [int(idx) for idx in state_dict.get("exclude_indices", self.exclude_indices)]
        self.rng.bit_generator.state = state_dict["rng_state"]

    def __len__(self):
        return len(self.dataset)
