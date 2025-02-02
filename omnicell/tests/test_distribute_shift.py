import numpy as np
import pytest
from ..models.distribute_shift import (
    get_proportional_weighted_dist,
    sample_multinomial_batch,
    sample_pert
)

class TestDistributeShift:
    @pytest.fixture
    def simple_matrix(self):
        return np.array([
            [1., 2., 0.],
            [3., 4., 0.],
            [2., 4., 0.]
        ], dtype=np.float32)

    def test_proportional_weighted_dist_basic(self, simple_matrix):
        result = get_proportional_weighted_dist(simple_matrix)
        
        # Check shape
        assert result.shape == simple_matrix.shape
        
        # Check columns sum to 1 (or 0 for zero columns)
        np.testing.assert_allclose(
            result.sum(axis=0),
            np.array([1., 1., 0.], dtype=np.float32),
            atol=1e-6  # Increased tolerance for float32
        )
        
        # Check proportions in first column
        expected_col0 = np.array([1/6, 3/6, 2/6], dtype=np.float32)
        np.testing.assert_allclose(result[:, 0], expected_col0, atol=1e-6)

    def test_proportional_weighted_dist_zero_handling(self):
        X = np.array([[0., 1.], [0., 2.]], dtype=np.float32)
        result = get_proportional_weighted_dist(X)
        
        # First column should be all zeros
        assert np.all(result[:, 0] == 0)
        # Second column should sum to 1
        assert np.abs(result[:, 1].sum() - 1.0) < 1e-6

    def test_sample_multinomial_batch_basic(self):
        probs = np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float32)
        counts = np.array([100, 200], dtype=np.float32)
        
        result = sample_multinomial_batch(probs, counts)
        
        # Check shape
        assert result.shape == probs.shape
        
        # Check counts sum correctly
        np.testing.assert_array_equal(
            result.sum(axis=0).astype(np.int32),
            counts.astype(np.int32)
        )
        
        # Check all values are non-negative
        assert np.all(result >= 0)

    def test_sample_multinomial_batch_with_max_count(self):
        probs = np.array([[0.3, 0.5], [0.7, 0.5]], dtype=np.float32)
        counts = np.array([100, 200], dtype=np.float32)
        max_count = np.array([[50, 150], [50, 150]], dtype=np.float32)
        
        result = sample_multinomial_batch(probs, counts, max_count)
        
        # Check no value exceeds max_count
        assert np.all(result <= max_count)
        
        # Check counts sum correctly 
        np.testing.assert_array_equal(
            result.sum(axis=0).astype(np.int32),
            counts.astype(np.int32)
        )
        
        # Check all values are non-negative
        assert np.all(result >= 0)

    def test_sample_pert_basic(self):
        ctrl = np.array([[10, 20], [30, 40]], dtype=np.float32)
        pert = np.array([[10, 10], [60, 40]], dtype=np.float32)
        mean_shift = pert.mean(axis=0) - ctrl.mean(axis=0)
        weighted_dist = np.array([[0.3, 0.4], [0.7, 0.6]], dtype=np.float32)
        
        result = sample_pert(ctrl, weighted_dist, mean_shift)
        
        # Check shape
        assert result.shape == ctrl.shape
        
        # Check all values are non-negative
        assert np.all(result >= 0)
        
        # Check direction of shifts
        assert np.mean(result[:, 0]) > np.mean(ctrl[:, 0])  # First column should increase
        assert np.mean(result[:, 1]) < np.mean(ctrl[:, 1])  # Second column should decrease

    def test_edge_cases(self):
        # Test empty matrix
        empty = np.array([[]], dtype=np.float32)
        result = get_proportional_weighted_dist(empty.reshape(0, 0))
        assert result.size == 0

        # Test single element
        single = np.array([[1.0]], dtype=np.float32)
        result = get_proportional_weighted_dist(single)
        assert result.item() == 1.0

        # Test all zeros
        zeros = np.zeros((3, 3), dtype=np.float32)
        result = get_proportional_weighted_dist(zeros)
        assert np.all(result == 0)

    def test_numerical_stability(self):
        # Test with very small numbers
        small = np.array([[1e-7, 1e-7], [2e-7, 3e-7]], dtype=np.float32)  # Adjusted for float32 range
        result = get_proportional_weighted_dist(small)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result.sum(axis=0), [1, 1], rtol=1e-6)

        # Test with very large numbers
        large = np.array([[1e7, 1e7], [2e7, 3e7]], dtype=np.float32)  # Adjusted for float32 range
        result = get_proportional_weighted_dist(large)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result.sum(axis=0), [1, 1], rtol=1e-6)