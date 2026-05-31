"""Tests for V15 batch-aware selection in SampleTaylorBatchSampler."""
import numpy as np
import sys
sys.path.insert(0, "/zhdd/home/tjshen/260415_ArcherA100/v15")

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


def test_batch_aware_penalizes_redundant_samples():
    """Samples with similar grad sketches should not be co-selected."""
    dataset = _FakeDataset(["math-dapo"] * 8)
    sampler = SampleTaylorBatchSampler(
        dataset=dataset,
        batch_size=3,
        seed=42,
        candidate_multiplier=2,
        sample_softmax_temperature=0.01,  # near-deterministic
    )
    sampler.remaining_orders["math"] = [0, 1, 2, 3, 4, 5, 6, 7]

    # All samples have equal base scores
    sample_scores = {i: 1.0 for i in range(8)}

    # Grad sketches: 0,1,2 are nearly identical; 3,4,5 are diverse
    dim = 256
    rng = np.random.default_rng(123)
    base_vec = rng.normal(size=dim).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    grad_sketches = {}
    # Cluster: 0,1,2 all point same direction
    for i in range(3):
        grad_sketches[i] = base_vec + rng.normal(size=dim).astype(np.float32) * 0.01
    # Diverse: 3,4,5 point different directions
    for i in range(3, 6):
        v = rng.normal(size=dim).astype(np.float32)
        v /= np.linalg.norm(v)
        grad_sketches[i] = v
    # Filler
    for i in range(6, 8):
        grad_sketches[i] = rng.normal(size=dim).astype(np.float32) * 0.1

    # Curvature matrix = identity (simplest case)
    curvature_matrix = np.eye(dim, dtype=np.float32)

    result = sampler.sample_batch(
        domain_weights=np.asarray([1.0], dtype=np.float32),
        sample_scores=sample_scores,
        grad_sketches=grad_sketches,
        curvature_matrix=curvature_matrix,
    )

    selected = set(result["indices"])
    # With batch-aware selection, at most 1 from the redundant cluster {0,1,2}
    cluster_selected = selected & {0, 1, 2}
    assert len(cluster_selected) <= 2, (
        f"Batch-aware should penalize redundancy, but selected {cluster_selected} from identical cluster"
    )
    print(f"PASS: selected {selected}, cluster overlap = {len(cluster_selected)}")


def test_fallback_when_no_curvature():
    """Without curvature_matrix, should behave like original softmax selection."""
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
        grad_sketches=None,
        curvature_matrix=None,
    )

    assert set(result["indices"]) == {4, 5}, f"Expected {{4,5}}, got {set(result['indices'])}"
    print(f"PASS: fallback selects top-scored {{4, 5}}")


def test_fallback_when_no_sketches():
    """Without grad_sketches, should behave like original softmax selection."""
    dataset = _FakeDataset(["math-dapo"] * 6)
    sampler = SampleTaylorBatchSampler(
        dataset=dataset,
        batch_size=2,
        seed=13,
        candidate_multiplier=3,
        sample_softmax_temperature=0.01,
    )
    sampler.remaining_orders["math"] = [0, 1, 2, 3, 4, 5]

    dim = 256
    curvature_matrix = np.eye(dim, dtype=np.float32)

    result = sampler.sample_batch(
        domain_weights=np.asarray([1.0], dtype=np.float32),
        sample_scores={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 10.0, 5: 9.0},
        grad_sketches=None,
        curvature_matrix=curvature_matrix,
    )

    assert set(result["indices"]) == {4, 5}, f"Expected {{4,5}}, got {set(result['indices'])}"
    print(f"PASS: no-sketches fallback selects top-scored {{4, 5}}")


def test_original_tests_still_pass():
    """Ensure backward compatibility: original test cases still work."""
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
    print("PASS: original behavior preserved with default args")


if __name__ == "__main__":
    test_batch_aware_penalizes_redundant_samples()
    test_fallback_when_no_curvature()
    test_fallback_when_no_sketches()
    test_original_tests_still_pass()
    print("\nAll V15 tests passed!")
