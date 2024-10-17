import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

# Equations taken from https://github.com/modAL-python/modAL

def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff) / std)


def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)


def UCB(mean, std, kappa):
    return mean + kappa * std

# Greedy sampling by maximising entropy of the updated covariance matrix with candidate compounds

def calculate_entropy(cov):
    """Calculate the entropy of a Gaussian distribution given the covariance matrix."""
    cov_det = np.linalg.det(cov)
    if cov_det <= 0:
        return 0  # Return zero if covariance is singular
    d = cov.shape[0]
    entropy = d/2 * np.log(2*np.pi) + 0.5 * np.log(cov_det) + 0.5 * d
    return entropy

def greedy_batch_selection(K_train_train, K_train_candidates, K_candidates_candidates, batch_size):
    """
    Perform greedy selection of candidate points to maximize entropy.
    
    Parameters:
        K_train_train: Covariance matrix of the training data (np.ndarray)
        K_train_candidates: Covariance between training and candidate points (np.ndarray)
        K_candidates_candidates: Covariance between the candidate points (np.ndarray)
        batch_size: Desired number of points to select for the batch
        
    Returns:
        selected_indices: Indices of selected points that maximize entropy
    """
    selected_indices = []
    remaining_candidates = list(range(K_candidates_candidates.shape[0]))

    for _ in range(batch_size):
        max_entropy = -float("inf")
        best_candidate_idx = None

        # Iterate over all remaining candidates (using the reduced matrix size)
        for idx in range(len(remaining_candidates)):

            # Find the current candidate's original index
            original_idx = remaining_candidates[idx]
            
            # Build the covariance matrix for this candidate
            candidate_cov = K_candidates_candidates[np.ix_([idx], [idx])]
            combined_cov = update_covariance_matrix(K_train_train, 
                                                    K_train_candidates[:, [idx]], 
                                                    candidate_cov)

            # Calculate entropy of the combined covariance matrix
            entropy = calculate_entropy(combined_cov)

            # Greedy selection: Choose the candidate with the maximum entropy
            if entropy > max_entropy:
                max_entropy = entropy
                best_candidate_idx = idx

        # Add the selected candidate (original index) to the batch
        selected_indices.append(remaining_candidates[best_candidate_idx])

        # Remove the best candidate from the remaining candidates
        remaining_candidates_idx = list(range(len(remaining_candidates)))
        _ = remaining_candidates_idx.pop(best_candidate_idx)
        remaining_candidates.pop(best_candidate_idx)

        # Update covariance matrices after selecting the best candidate
        best_candidate_cov = K_candidates_candidates[np.ix_([best_candidate_idx], [best_candidate_idx])]
        K_train_candidates_best = K_train_candidates[:, [best_candidate_idx]]
        
        K_train_train = update_covariance_matrix(K_train_train, K_train_candidates_best, best_candidate_cov)

        K_train_candidates = np.delete(K_train_candidates, best_candidate_idx, axis=1)
        new_row = K_candidates_candidates[best_candidate_idx, remaining_candidates_idx].reshape(1, -1)
        K_train_candidates = np.vstack([K_train_candidates, new_row])

        K_candidates_candidates = np.delete(K_candidates_candidates, best_candidate_idx, axis=0)
        K_candidates_candidates = np.delete(K_candidates_candidates, best_candidate_idx, axis=1)

    return selected_indices


def update_covariance_matrix(K_train_train, K_train_candidate, candidate_cov):
    """
    Update the covariance matrix after selecting a candidate point.
    
    Parameters:
        K_train_train: Current covariance matrix for training data (np.ndarray)
        K_train_candidate: Cross-covariance between training and candidate (np.ndarray)
        candidate_cov: Covariance of the candidate point (np.ndarray)
    
    Returns:
        Updated covariance matrix (np.ndarray)
    """
    # Stack arrays to form the new covariance matrix
    top = np.hstack((K_train_train, K_train_candidate))
    bottom = np.hstack((K_train_candidate.T, candidate_cov))
    updated_cov = np.vstack((top, bottom))
    
    return updated_cov


## Acquisition functions from ___ gPO acquisition function
from scipy.stats import multivariate_normal
def acquire_gPO(mean, cov, c: int = 1, batch_size: int = 100, Ns: int = 10000, seed: int = None, **kwargs): 
    """ The proposed acquisition function -- qPO (multipoint probability of optimality) """
    
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=Ns, random_state=seed)
    except: 
        count = 0
        sampled = False 
        while count < 10 and not sampled: 
            print('Error sampling from multivariate, adding noise to diagonal')
            try: 
                cov = cov + np.identity(len(mean))*1e-8
                p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
                samples = p_yx.rvs(size=Ns, random_state=seed)
                sampled = True 
            except: 
                continue
    
    top_samples = np.array([np.argmax(c*sample) for sample in samples])
    probs = np.bincount(top_samples, minlength=len(mean))/Ns # [np.sum(top_k_samples==i)/N_samples for i in range(samples.shape[1])]

    return np.argsort(probs)[::-1][:10], np.array(probs)[np.argsort(probs)[::-1][:10]]

def acquire_ts(mean, cov, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs): 
    """ Acquisition with parallel Thomspon sampling """

    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=batch_size, random_state=seed)
    except: 
        count = 0
        sampled = False 
        while count < 10 and not sampled:
            try:
                print('Error sampling from multivariate, adding noise to diagonal')
                cov = cov + np.identity(len(mean))*1e-8
                p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
                samples = p_yx.rvs(size=batch_size, random_state=seed)
                sampled = True 
            except: 
                continue

    selected_inds = []

    for sample in samples:
        for ind in np.argsort(c*sample)[::-1]:            
            if ind not in selected_inds: 
                selected_inds.append(ind)
                break 

    return selected_inds