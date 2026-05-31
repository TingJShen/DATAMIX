import numpy as np

from dapo.sample_taylor_sampler import SampleTaylorBatchSampler


class _FakeIloc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return self._rows[idx]


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = _FakeIloc(rows)

    def __len__(self):
        return len(self._rows)


class _FakeDataset:
    def __init__(self, sources):
        self.dataframe = _FakeDataFrame([{"data_source": source} for source in sources])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]


def test_sample_taylor_sampler_selects_high_scored_candidates_without_consuming_unselected():
    dataset = _FakeDataset(["math-dapo"] * 6)
    sampler = SampleTaylorBatchSampler(
        dataset=dataset,
        batch_size=2,
        seed=13,
        candidate_multiplier=3,
        sample_softmax_temperature=0.01,
    )
    sampler.remaining_orders["math"] = [0, 1, 2, 3, 4, 5]

    result = sampler.sample_batch(
        domain_weights=np.asarray([1.0], dtype=np.float32),
        sample_scores={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.0, 5: 9.0},
    )

    assert set(result["indices"]) == {4, 5}
    assert sampler.remaining_orders["math"] == [0, 1, 2, 3]
    assert result["candidate_counts"]["math"] == 6
    assert result["score_means"]["math"] > 3.0


def test_sample_taylor_sampler_state_round_trip_preserves_remaining_pool_and_rng():
    dataset = _FakeDataset(["math-dapo"] * 4 + ["code-dapo"] * 4)
    sampler = SampleTaylorBatchSampler(dataset=dataset, batch_size=4, seed=29)
    sampler.remaining_orders["math"] = [0, 1]
    sampler.remaining_orders["code"] = [4, 5, 6]
    state = sampler.state_dict()

    restored = SampleTaylorBatchSampler(dataset=dataset, batch_size=4, seed=1)
    restored.load_state_dict(state)

    assert restored.remaining_orders == sampler.remaining_orders
    assert restored.reset_counts == sampler.reset_counts
    assert restored.total_drawn == sampler.total_drawn
    assert restored.rng.bit_generator.state == sampler.rng.bit_generator.state


def test_sample_taylor_sampler_enforces_domain_min_weight_before_count_allocation():
    dataset = _FakeDataset(["math-dapo"] * 5 + ["code-dapo"] * 5 + ["wildchat"] * 5)
    sampler = SampleTaylorBatchSampler(
        dataset=dataset,
        batch_size=6,
        seed=7,
        domain_min_weight=0.2,
    )

    result = sampler.sample_batch(
        domain_weights=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        sample_scores={idx: 1.0 for idx in range(len(dataset))},
    )

    assert result["target_counts"]["math"] >= 2
    assert result["target_counts"]["code"] >= 1
    assert result["target_counts"]["general"] >= 1
    assert sum(result["target_counts"].values()) == 6


def test_peek_candidates_does_not_consume_remaining_pool():
    dataset = _FakeDataset(["math-dapo"] * 6)
    sampler = SampleTaylorBatchSampler(dataset=dataset, batch_size=2, seed=5, candidate_multiplier=2)
    sampler.remaining_orders["math"] = [0, 1, 2, 3, 4, 5]

    info = sampler.peek_candidates(np.asarray([1.0], dtype=np.float32))

    assert info["candidate_indices_by_category"]["math"] == [0, 1, 2, 3]
    assert info["target_counts"]["math"] == 2
    assert sampler.remaining_orders["math"] == [0, 1, 2, 3, 4, 5]


def test_excluded_indices_never_enter_training_pool():
    dataset = _FakeDataset(["math-dapo"] * 6)
    sampler = SampleTaylorBatchSampler(
        dataset=dataset,
        batch_size=2,
        seed=5,
        exclude_indices=[0, 1],
    )

    assert set(sampler.category_indices["math"].tolist()) == {2, 3, 4, 5}
    result = sampler.sample_batch(
        domain_weights=np.asarray([1.0], dtype=np.float32),
        sample_scores={idx: 1.0 for idx in range(6)},
    )
    assert set(result["indices"]).isdisjoint({0, 1})
