"""
Re-ranking stage for the writer identification retrieval pipeline.

Provides two complementary, training-free steps that operate on the final
encodings / distance matrix:

  * alpha query expansion (alpha-QE), Radenovic et al., "Fine-tuning CNN Image
    Retrieval with No Human Annotation"
  * k-reciprocal encoding with Jaccard distance, Zhong et al., CVPR 2017

Both assume a symmetric setting where the query set is also the gallery, which
is exactly the leave-one-image-out protocol used for ICDAR17 writer retrieval.
"""

import numpy as np

EPS = 1e-12


def l2norm_rows(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + EPS)


def cosine_distance_matrix(encs):
    """
    raw cosine distance matrix (self-distance stays ~0, needed by re-ranking)
    parameters:
        encs: NxD encoding matrix
    returns: NxN distance matrix in [0, 2]
    """
    encs = l2norm_rows(encs)
    dists = 1.0 - np.dot(encs, encs.T)
    np.fill_diagonal(dists, 0.0)
    return np.clip(dists, 0.0, None).astype(np.float32)


def mask_self_distances(dists):
    """ return a copy where retrieving the query itself is impossible """
    dists = np.array(dists, dtype=np.float32, copy=True)
    np.fill_diagonal(dists, np.finfo(np.float32).max)
    return dists


def alpha_query_expansion(encs, k=5, alpha=3.0, n_iter=1):
    """
    alpha-weighted query expansion / database-side augmentation
    parameters:
        encs: NxD encoding matrix
        k: number of neighbors aggregated into each query
        alpha: exponent on the similarity used as neighbor weight
        n_iter: how often the expansion is repeated
    returns: NxD matrix of expanded, l2-normalized encodings
    """
    encs = l2norm_rows(encs)
    if k <= 0:
        return encs

    expanded = encs
    n = encs.shape[0]
    k = min(k, n - 1)
    for _ in range(max(1, n_iter)):
        sims = np.dot(expanded, expanded.T)
        np.fill_diagonal(sims, -np.inf)
        # top-k neighbors per row
        nn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        nn_sims = sims[rows, nn_idx]
        order = np.argsort(-nn_sims, axis=1)
        nn_idx = nn_idx[rows, order]
        nn_sims = nn_sims[rows, order]

        weights = np.clip(nn_sims, 0.0, None) ** alpha
        # query itself always keeps weight 1
        neighbors = (weights[:, :, None] * expanded[nn_idx]).sum(axis=1)
        expanded = l2norm_rows(encs + neighbors)

    return expanded


def k_reciprocal_rerank(orig_dist, k1=20, k2=6, lambda_value=0.3, verbose=False):
    """
    k-reciprocal encoding re-ranking (Zhong et al., CVPR 2017)
    parameters:
        orig_dist: NxN distance matrix with unmasked self-distances
        k1: neighborhood size used to build the k-reciprocal sets
        k2: neighborhood size of the local query expansion on the sparse
            k-reciprocal features (k2=1 disables it)
        lambda_value: weight of the original distance, 0 = pure Jaccard
    returns: NxN re-ranked distance matrix
    """
    orig_dist = np.asarray(orig_dist, dtype=np.float32)
    all_num = orig_dist.shape[0]
    k1 = max(1, min(k1, all_num - 1))
    k2 = max(1, min(k2, all_num))

    # scale per column and symmetrize the way the reference implementation does
    col_max = np.max(orig_dist, axis=0)
    col_max[col_max < EPS] = 1.0
    norm_dist = np.transpose(orig_dist / col_max)

    initial_rank = np.argsort(norm_dist, axis=1).astype(np.int32)
    V = np.zeros((all_num, all_num), dtype=np.float32)
    half_k1 = int(np.around(k1 / 2.0)) + 1

    for i in range(all_num):
        forward = initial_rank[i, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        reciprocal = forward[np.where(backward == i)[0]]

        expansion = reciprocal
        for candidate in reciprocal:
            cand_forward = initial_rank[candidate, :half_k1]
            cand_backward = initial_rank[cand_forward, :half_k1]
            cand_reciprocal = cand_forward[np.where(cand_backward == candidate)[0]]
            # only merge neighborhoods that overlap strongly
            if len(np.intersect1d(cand_reciprocal, reciprocal)) > 2.0 / 3.0 * len(cand_reciprocal):
                expansion = np.append(expansion, cand_reciprocal)

        expansion = np.unique(expansion)
        weight = np.exp(-norm_dist[i, expansion])
        V[i, expansion] = weight / (np.sum(weight) + EPS)

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros_like(norm_dist)
    for i in range(all_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        ind_non_zero = np.where(V[i, :] != 0)[0]
        for ind in ind_non_zero:
            images = inv_index[ind]
            np.add.at(temp_min, images, np.minimum(V[i, ind], V[images, ind]))
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min + EPS)
        if verbose and (i + 1) % 500 == 0:
            print('  k-reciprocal: {}/{}'.format(i + 1, all_num))

    final_dist = jaccard_dist * (1.0 - lambda_value) + norm_dist * lambda_value
    np.fill_diagonal(final_dist, 0.0)
    return final_dist.astype(np.float32)


def rerank_distances(encs, method='qe+kreciprocal', qe_k=2, qe_alpha=3.0,
                     qe_iter=1, k1=4, k2=2, lambda_value=0.3, verbose=True):
    """
    run the re-ranking stage and return a distance matrix ready for evaluation
    parameters:
        encs: NxD encoding matrix
        method: one of 'none', 'qe', 'kreciprocal', 'qe+kreciprocal'
    returns: NxN distance matrix with masked self-distances
    """
    valid = ('none', 'qe', 'kreciprocal', 'qe+kreciprocal')
    if method not in valid:
        raise ValueError('unknown re-ranking method {!r}, expected one of {}'.format(method, valid))

    encs = l2norm_rows(encs)
    if method in ('qe', 'qe+kreciprocal'):
        if verbose:
            print('> re-ranking: alpha-QE (k={}, alpha={}, iter={})'.format(qe_k, qe_alpha, qe_iter))
        encs = alpha_query_expansion(encs, k=qe_k, alpha=qe_alpha, n_iter=qe_iter)

    dists = cosine_distance_matrix(encs)
    if method in ('kreciprocal', 'qe+kreciprocal'):
        if verbose:
            print('> re-ranking: k-reciprocal (k1={}, k2={}, lambda={})'.format(k1, k2, lambda_value))
        dists = k_reciprocal_rerank(dists, k1=k1, k2=k2, lambda_value=lambda_value,
                                    verbose=verbose)

    return mask_self_distances(dists)
