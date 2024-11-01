import pytest
import numpy as np
from scipy.stats import multivariate_normal
from gp_qsar.acquisition_functions import(
    greedy_batch_selection,
    acquire_gPO,
    acquire_ts,
)


def test_acquire_ts_successful_sampling():
    # Test with mean and cov larger than batch_size
    mean = np.array([0, 1, 2, 3, 4])
    cov = np.eye(len(mean))  # identity matrix for simplicity
    batch_size = 3
    selected_inds = acquire_ts(mean, cov, c=1, batch_size=batch_size, seed=42)
    
    assert (all(0 <= ind < len(mean) for ind in selected_inds))
    assert len(selected_inds) == batch_size


def test_acquire_ts_with_singular_matrix():
    # Test with a nearly singular covariance matrix to simulate a failure in rvs
    mean = np.array([1, 2, 3,4,5,6])
    cov = np.ones((len(mean),len(mean)))
    batch_size = 3

    # Run the function and expect it to handle the error gracefully
    selected_inds = acquire_ts(mean, cov, c=1, batch_size=batch_size, seed=42)
    
    # Verify that the output is as expected despite initial sampling failure
    assert (all(0 <= ind < len(mean) for ind in selected_inds))
    assert len(selected_inds) == batch_size


def test_acquire_gpo_successful_sampling():
    # Test with mean and cov larger than batch_size
    mean = np.array([0, 1, 2, 3, 4])
    cov = np.eye(len(mean))  # identity matrix for simplicity
    batch_size = 3
    selected_inds, probs = acquire_gPO(mean, cov, c=1, batch_size=batch_size, seed=42)
    
    assert (all(0 <= ind < len(mean) for ind in selected_inds))
    assert len(selected_inds) == batch_size


def test_acquire_gpo_with_singular_matrix():
    # Test with a singular covariance matrix to (ideally) simulate a failure in rvs
    mean = np.array([1, 2, 3,4,5,6])
    cov = np.ones((len(mean),len(mean)))
    batch_size = 3

    # Run the function and expect it to handle the error gracefully
    # Note - does not actually break the process - Multivariate norm in scipy can handle singular matrix
    selected_inds, probs = acquire_gPO(mean, cov, c=1, batch_size=batch_size, seed=42)
    
    # Verify that the output is as expected despite initial sampling failure
    assert (all(0 <= ind < len(mean) for ind in selected_inds))
    assert len(selected_inds) == batch_size

