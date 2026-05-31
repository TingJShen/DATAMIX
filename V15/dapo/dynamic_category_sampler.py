#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic Category Sampler for DAPO Training
根据训练过程中的reward动态调整各类别的采样权重
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sized

import numpy as np
import torch
from torch.utils.data import Sampler


class DynamicCategorySampler(Sampler):
    """
    动态类别采样器：根据各类别的reward表现动态调整采样权重

    特点：
    1. 支持按类别索引数据
    2. 根据reward提升率动态调整权重
    3. 支持训练过程中实时更新权重
    """

    def __init__(
        self,
        data_source: Sized,
        category_indices: Dict[str, List[int]],
        initial_weights: Optional[Dict[str, float]] = None,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        smooth_factor: float = 0.3,
        generator: Optional[torch.Generator] = None
    ):
        """
        Args:
            data_source: 数据集
            category_indices: 类别到索引的映射，如 {'math': [0,1,2], 'code': [3,4,5], 'general': [6,7,8]}
            initial_weights: 初始采样权重
            min_weight: 单个类别的最小权重
            max_weight: 单个类别的最大权重
            smooth_factor: 权重平滑因子（新旧权重混合）
            generator: 随机数生成器
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.category_indices = category_indices
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smooth_factor = smooth_factor
        self.generator = generator

        # 类别列表
        self.categories = list(category_indices.keys())
        self.num_categories = len(self.categories)

        # 初始化权重
        if initial_weights is None:
            initial_weight = 1.0 / self.num_categories
            self.weights = {cat: initial_weight for cat in self.categories}
        else:
            self.weights = initial_weights.copy()

        # 归一化权重
        self._normalize_weights()

        # 记录每个类别的采样数量
        self.samples_per_category = {cat: len(indices) for cat, indices in category_indices.items()}
        self.total_samples = len(data_source)

        print(f"DynamicCategorySampler initialized:")
        print(f"  Categories: {self.categories}")
        print(f"  Samples per category: {self.samples_per_category}")
        print(f"  Initial weights: {self.weights}")

    def _normalize_weights(self):
        """归一化权重使其和为1"""
        total = sum(self.weights.values())
        if total > 0:
            for cat in self.weights:
                self.weights[cat] /= total

    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新采样权重（带平滑）

        Args:
            new_weights: 新的权重字典
        """
        # 应用约束
        for cat in self.categories:
            if cat in new_weights:
                # 限制范围
                w = max(self.min_weight, min(self.max_weight, new_weights[cat]))
                # 平滑更新
                self.weights[cat] = (1 - self.smooth_factor) * self.weights[cat] + self.smooth_factor * w

        # 归一化
        self._normalize_weights()

        print(f"Updated sampling weights: {self.weights}")

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

    def __iter__(self):
        """
        生成一个epoch的索引序列

        策略：根据权重确定每个类别的采样数量，然后从每个类别中随机采样
        """
        indices = []

        # 计算每个类别应该采样的数量
        for cat in self.categories:
            cat_indices = self.category_indices[cat]
            weight = self.weights[cat]

            # 按权重比例采样
            n_samples = int(self.total_samples * weight)
            n_samples = max(1, min(n_samples, len(cat_indices)))  # 至少采样1个，最多采样该类别全部

            # 随机采样（有放回，确保足够样本）
            if n_samples > len(cat_indices):
                # 需要重复采样
                sampled = np.random.choice(cat_indices, size=n_samples, replace=True)
            else:
                sampled = np.random.choice(cat_indices, size=n_samples, replace=False)

            indices.extend(sampled)

        # 打乱顺序
        np.random.shuffle(indices)

        # 迭代生成索引
        for idx in indices:
            yield idx

    def __len__(self):
        return self.total_samples


class GlobalCategoryPoolSampler:
    """
    全局类别采样器：直接从整个训练集按类别抽样。

    规则：
    1. 每个类别维护一份打乱后的“未抽取”索引池。
    2. 同一类别在索引池耗尽前不重复使用样本。
    3. 某类别索引池耗尽后，仅重置该类别的索引池并重新打乱。
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 1,
        categories: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        raw_category_indices = build_category_indices(dataset)
        preferred_categories = categories or ["math", "code", "general"]
        self.categories = [cat for cat in preferred_categories if raw_category_indices.get(cat)]
        self.category_indices = {
            cat: np.asarray(raw_category_indices[cat], dtype=np.int64) for cat in self.categories
        }
        self.positions = {cat: 0 for cat in self.categories}
        self.permutations: Dict[str, np.ndarray] = {}
        self.reset_counts = {cat: 0 for cat in self.categories}
        self.total_drawn = {cat: 0 for cat in self.categories}

        for cat in self.categories:
            self._reset_category(cat, initial=True)

        samples_per_category = {cat: int(len(indices)) for cat, indices in self.category_indices.items()}
        print("GlobalCategoryPoolSampler initialized:")
        print(f"  Categories: {self.categories}")
        print(f"  Samples per category: {samples_per_category}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Seed: {self.seed}")

    def _reset_category(self, category: str, initial: bool = False):
        indices = self.category_indices[category]
        if len(indices) == 0:
            self.permutations[category] = np.asarray([], dtype=np.int64)
            self.positions[category] = 0
            return

        self.permutations[category] = self.rng.permutation(indices)
        self.positions[category] = 0
        if not initial:
            self.reset_counts[category] += 1

    def _normalize_weights(self, current_weights: Any) -> Dict[str, float]:
        if isinstance(current_weights, dict):
            weight_map = {cat: float(current_weights.get(cat, 0.0)) for cat in self.categories}
        else:
            weight_values = list(current_weights)
            if len(weight_values) < len(self.categories):
                raise ValueError(
                    f"Expected at least {len(self.categories)} weights, got {len(weight_values)}"
                )
            weight_map = {cat: float(weight_values[i]) for i, cat in enumerate(self.categories)}

        total = sum(max(weight, 0.0) for weight in weight_map.values())
        if total <= 0:
            uniform_weight = 1.0 / len(self.categories)
            return {cat: uniform_weight for cat in self.categories}
        return {cat: max(weight_map[cat], 0.0) / total for cat in self.categories}

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

    def _draw_from_category(self, category: str, target_num: int) -> tuple[list[int], int]:
        selected: list[int] = []
        reset_times = 0

        if target_num <= 0:
            return selected, reset_times
        if len(self.category_indices[category]) == 0:
            raise ValueError(f"Category {category} has no available samples.")

        while len(selected) < target_num:
            available = len(self.permutations[category]) - self.positions[category]
            if available <= 0:
                self._reset_category(category)
                reset_times += 1
                available = len(self.permutations[category]) - self.positions[category]

            take = min(target_num - len(selected), available)
            start = self.positions[category]
            end = start + take
            selected.extend(self.permutations[category][start:end].tolist())
            self.positions[category] = end

        self.total_drawn[category] += len(selected)
        return selected, reset_times

    def sample_batch(self, current_weights: Any) -> Dict[str, Any]:
        normalized_weights = self._normalize_weights(current_weights)
        target_counts = self._allocate_target_counts(normalized_weights)

        sampled_indices: list[int] = []
        sampled_counts = {cat: 0 for cat in self.categories}
        reset_events = {cat: 0 for cat in self.categories}

        for cat in self.categories:
            chosen, reset_times = self._draw_from_category(cat, target_counts[cat])
            sampled_indices.extend(chosen)
            sampled_counts[cat] = len(chosen)
            reset_events[cat] = reset_times

        sampled_indices = np.asarray(sampled_indices, dtype=np.int64)
        self.rng.shuffle(sampled_indices)

        ratios = {
            cat: float(sampled_counts[cat]) / float(self.batch_size) for cat in self.categories
        }
        remaining = {
            cat: int(len(self.permutations[cat]) - self.positions[cat]) for cat in self.categories
        }
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
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seed": self.seed,
            "categories": list(self.categories),
            "category_indices": {
                cat: indices.tolist() for cat, indices in self.category_indices.items()
            },
            "positions": dict(self.positions),
            "permutations": {
                cat: perm.tolist() for cat, perm in self.permutations.items()
            },
            "reset_counts": dict(self.reset_counts),
            "total_drawn": dict(self.total_drawn),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        saved_categories = list(state_dict.get("categories", []))
        if saved_categories != self.categories:
            raise ValueError(
                f"Category mismatch when loading sampler state: {saved_categories} != {self.categories}"
            )

        self.positions = {
            cat: int(state_dict["positions"][cat]) for cat in self.categories
        }
        self.permutations = {
            cat: np.asarray(state_dict["permutations"][cat], dtype=np.int64)
            for cat in self.categories
        }
        self.reset_counts = {
            cat: int(state_dict["reset_counts"][cat]) for cat in self.categories
        }
        self.total_drawn = {
            cat: int(state_dict.get("total_drawn", {}).get(cat, 0)) for cat in self.categories
        }
        self.rng.bit_generator.state = state_dict["rng_state"]

    def __len__(self):
        return len(self.dataset)


def build_category_indices(dataset) -> Dict[str, List[int]]:
    """
    从数据集构建类别索引

    Args:
        dataset: RLHFDataset数据集

    Returns:
        类别到索引的映射
    """
    category_indices = defaultdict(list)

    # 遍历数据集
    dataframe = dataset.dataframe

    for idx in range(len(dataframe)):
        row = dataframe.iloc[idx]
        data_source = row.get('data_source', 'unknown')

        # 根据data_source确定类别
        if data_source.startswith('math'):
            category = 'math'
        elif data_source.startswith('code'):
            category = 'code'
        elif data_source.startswith('wildchat'):
            category = 'general'
        else:
            category = 'other'

        category_indices[category].append(idx)

    # 转换为普通dict
    return dict(category_indices)


if __name__ == "__main__":
    # 测试代码
    print("Testing DynamicCategorySampler...")

    # 模拟数据
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size

    # 创建类别索引
    category_indices = {
        'math': list(range(0, 40)),
        'code': list(range(40, 75)),
        'general': list(range(75, 100))
    }

    dataset = MockDataset(100)
    sampler = DynamicCategorySampler(
        data_source=dataset,
        category_indices=category_indices,
        min_weight=0.15,
        max_weight=0.5
    )

    # 测试迭代
    indices = list(sampler)
    print(f"Generated {len(indices)} indices")

    # 统计各类别采样数量
    cat_counts = defaultdict(int)
    for idx in indices:
        for cat, cat_idx in category_indices.items():
            if idx in cat_idx:
                cat_counts[cat] += 1
                break
    print(f"Samples per category: {dict(cat_counts)}")

    # 测试权重更新
    print("\nUpdating weights...")
    sampler.update_weights({'math': 0.5, 'code': 0.3, 'general': 0.2})

    indices = list(sampler)
    cat_counts = defaultdict(int)
    for idx in indices:
        for cat, cat_idx in category_indices.items():
            if idx in cat_idx:
                cat_counts[cat] += 1
                break
    print(f"Samples per category after update: {dict(cat_counts)}")

    print("\nTest passed!")
