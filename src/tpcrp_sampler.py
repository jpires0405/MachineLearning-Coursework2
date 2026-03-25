"""
TPC_RP Sampler — TypiClust with Representation learning + Parametric clustering

Implements Algorithm 1 from:
    Hacohen, G., Dekel, A., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022.

Selection pipeline
------------------
1. Cluster all embeddings with K-Means (K = min(|L| + B, 500)).
2. Identify uncovered clusters (no currently-labeled point inside) and
   discard clusters with fewer than MIN_CLUSTER_SIZE total points.
3. Sort valid uncovered clusters by size (descending); pick the top-B
   largest.  If fewer than B are available, fall back to random sampling
   from the remaining unlabeled pool to reach the budget.
4. Within each selected cluster pick the unlabeled point with the highest
   Typicality score (inverse of the mean L2 distance to its K_nn nearest
   neighbors, K_nn = min(20, cluster_size)).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def random_query(
    unlabeled_indices: list[int],
    budget: int,
    seed: int = 42,
) -> list[int]:
    """
    Uniform random baseline: select `budget` indices without replacement.

    Parameters
    ----------
    unlabeled_indices : list[int]
        Pool of indices not yet labeled.
    budget : int
        Number of examples to select.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list[int]  — exactly `budget` randomly chosen indices.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unlabeled_indices, size=min(budget, len(unlabeled_indices)),
                        replace=False)
    return chosen.tolist()

# Minimum cluster size to be considered a valid selection candidate
MIN_CLUSTER_SIZE = 5

# Default number of nearest neighbours for typicality
K_TYPICALITY = 20

# Hard cap on total number of K-Means clusters
MAX_CLUSTERS = 500


def compute_typicality(features: np.ndarray, k_nn: int) -> np.ndarray:
    """
    Compute the Typicality score for every point in `features`.

    Typicality(x) = ( mean_{x_i in K-NN(x)} ||x - x_i||_2 )^{-1}

    Parameters
    ----------
    features : np.ndarray  shape (N, D)
        L2-normalised embeddings for the points in one cluster.
    k_nn : int
        Number of nearest neighbours to use (excluding the point itself).

    Returns
    -------
    np.ndarray  shape (N,)  — typicality score per point (higher = more typical)
    """
    n = len(features)
    # We query k_nn + 1 neighbours so that we can drop the self-match (dist=0).
    # Cap at n so sklearn never requests more neighbours than samples.
    n_query = min(k_nn + 1, n)

    nbrs = NearestNeighbors(n_neighbors=n_query, algorithm='auto', metric='euclidean')
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)

    # distances[:, 0] is the self-distance (≈0); skip it.
    # If n_query == 1 (degenerate singleton), distances has shape (N, 1) and
    # distances[:, 0] is already self → mean over an empty slice → assign 0.
    if n_query == 1:
        # Only one point in the cluster; assign the same score to everyone.
        return np.ones(n, dtype=np.float32)

    neighbour_distances = distances[:, 1:]          # (N, k_nn)
    mean_dist = neighbour_distances.mean(axis=1)    # (N,)

    # Guard against zero-distance duplicates
    mean_dist = np.where(mean_dist == 0, 1e-10, mean_dist)

    return 1.0 / mean_dist


def tpcrp_query(
    all_features: np.ndarray,
    unlabeled_indices: list[int],
    labeled_indices: list[int],
    budget: int,
    seed: int = 42,
) -> list[int]:
    """
    Select `budget` examples from `unlabeled_indices` using TPC_RP.

    Parameters
    ----------
    all_features : np.ndarray  shape (N_total, 512)
        L2-normalised embeddings for *every* training example
        (indexed by the dataset's original integer index).
    unlabeled_indices : list[int]
        Indices of examples not yet labeled.
    labeled_indices : list[int]
        Indices of examples already labeled.
    budget : int
        Number of examples to query (B in the paper).
    seed : int
        Random seed for KMeans and any fallback random sampling.

    Returns
    -------
    list[int]
        Exactly `budget` dataset indices selected from `unlabeled_indices`.
    """
    rng = np.random.default_rng(seed)
    unlabeled_set = set(unlabeled_indices)
    labeled_set   = set(labeled_indices)

    # ── Step 1: K-Means clustering ────────────────────────────────────────────
    n_clusters = min(len(labeled_indices) + budget, MAX_CLUSTERS)
    # Edge case: if the unlabeled pool is smaller than n_clusters
    n_clusters = min(n_clusters, len(unlabeled_indices))

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(all_features)
    # cluster_labels[i] gives the cluster id for dataset index i

    # Build a mapping: cluster_id → list of global dataset indices
    cluster_to_indices: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for idx in range(len(all_features)):
        cluster_to_indices[cluster_labels[idx]].append(idx)

    # ── Step 2: Filter uncovered clusters ────────────────────────────────────
    valid_clusters: list[int] = []
    for c, members in cluster_to_indices.items():
        # Uncovered: no labeled point inside
        if any(idx in labeled_set for idx in members):
            continue
        # Must have enough total members to compute typicality reliably
        if len(members) < MIN_CLUSTER_SIZE:
            continue
        valid_clusters.append(c)

    # ── Step 3: Select top-B clusters by size ─────────────────────────────────
    valid_clusters.sort(key=lambda c: len(cluster_to_indices[c]), reverse=True)
    top_clusters = valid_clusters[:budget]

    # Collect already-selected indices to avoid duplicates during fallback
    selected: list[int] = []

    # ── Step 4: Typicality-based selection within each cluster ────────────────
    for c in top_clusters:
        members = cluster_to_indices[c]
        cluster_size = len(members)
        k_nn = min(K_TYPICALITY, cluster_size)

        # We only score unlabeled members
        unlabeled_members = [idx for idx in members if idx in unlabeled_set]
        if not unlabeled_members:
            continue

        features_cluster = all_features[unlabeled_members]           # (M, 512)
        typicality_scores = compute_typicality(features_cluster, k_nn)

        best_local = int(np.argmax(typicality_scores))
        selected.append(unlabeled_members[best_local])

    # ── Fallback: pad to budget with random unlabeled points ─────────────────
    if len(selected) < budget:
        selected_set = set(selected)
        remaining = [
            idx for idx in unlabeled_indices
            if idx not in selected_set
        ]
        shortfall = budget - len(selected)
        shortfall = min(shortfall, len(remaining))
        fallback = rng.choice(remaining, size=shortfall, replace=False).tolist()
        selected += fallback

    return selected[:budget]


def dynamic_tpcrp_query(
    all_features: np.ndarray,
    unlabeled_indices: list[int],
    labeled_indices: list[int],
    budget: int,
    phase_threshold: int = 60,
    seed: int = 42,
) -> list[int]:
    """
    Dynamic Phase-Shift TypiClust (DynTPC).

    Identical to tpcrp_query in Steps 1–3 (clustering, filtering, top-B
    cluster selection).  Step 4 switches the selection criterion based on
    the current labelled-set size relative to `phase_threshold`:

    - ``len(labeled_indices) < phase_threshold``  →  **argmax** typicality
      (most typical / densest point).  Cold-start regime: build a
      representative foundation quickly.
    - ``len(labeled_indices) >= phase_threshold`` →  **argmin** typicality
      (least typical / most atypical point).  High-budget regime: seek
      hard, ambiguous examples at cluster peripheries for discriminative
      refinement, as predicted by the phase-transition theory of
      Hacohen et al. (2022).

    Parameters
    ----------
    all_features : np.ndarray  shape (N_total, 512)
    unlabeled_indices : list[int]
    labeled_indices : list[int]
    budget : int
    phase_threshold : int
        Cumulative number of labeled examples at which the strategy flips
        from typicality-maximising to typicality-minimising.
        Default: 60 (= 6 × B for B=10, midpoint of the 10→100 range).
    seed : int

    Returns
    -------
    list[int]  — exactly `budget` dataset indices.
    """
    rng = np.random.default_rng(seed)
    unlabeled_set = set(unlabeled_indices)
    labeled_set   = set(labeled_indices)

    # Determine selection mode before clustering
    use_argmax = len(labeled_indices) < phase_threshold

    # ── Step 1: K-Means clustering ────────────────────────────────────────────
    n_clusters = min(len(labeled_indices) + budget, MAX_CLUSTERS)
    n_clusters = min(n_clusters, len(unlabeled_indices))

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(all_features)

    cluster_to_indices: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for idx in range(len(all_features)):
        cluster_to_indices[cluster_labels[idx]].append(idx)

    # ── Step 2: Filter uncovered clusters ────────────────────────────────────
    valid_clusters: list[int] = []
    for c, members in cluster_to_indices.items():
        if any(idx in labeled_set for idx in members):
            continue
        if len(members) < MIN_CLUSTER_SIZE:
            continue
        valid_clusters.append(c)

    # ── Step 3: Select top-B clusters by size ─────────────────────────────────
    valid_clusters.sort(key=lambda c: len(cluster_to_indices[c]), reverse=True)
    top_clusters = valid_clusters[:budget]

    selected: list[int] = []

    # ── Step 4: Phase-aware selection within each cluster ─────────────────────
    for c in top_clusters:
        members = cluster_to_indices[c]
        cluster_size = len(members)
        k_nn = min(K_TYPICALITY, cluster_size)

        unlabeled_members = [idx for idx in members if idx in unlabeled_set]
        if not unlabeled_members:
            continue

        features_cluster  = all_features[unlabeled_members]
        typicality_scores = compute_typicality(features_cluster, k_nn)

        # Phase switch: argmax in cold-start, argmin in high-budget
        chosen_local = int(np.argmax(typicality_scores) if use_argmax
                           else np.argmin(typicality_scores))
        selected.append(unlabeled_members[chosen_local])

    # ── Fallback ──────────────────────────────────────────────────────────────
    if len(selected) < budget:
        selected_set = set(selected)
        remaining = [i for i in unlabeled_indices if i not in selected_set]
        shortfall = min(budget - len(selected), len(remaining))
        selected += rng.choice(remaining, size=shortfall, replace=False).tolist()

    return selected[:budget]
