"""Device-agnostic k-means clustering in pure PyTorch (CPU / Apple-Silicon MPS).

This module replaces the previous dependency on ``torchpq.clustering.KMeans``.
``torchpq`` was imported at module scope by :mod:`ffsplat.models.transformations`
and pulls in ``cupy-cuda12x``, so *importing* ffsplat required a working CUDA
toolkit and the SOG codebook k-means could never run on a Mac.

The implementation here is float32-only, runs on whatever device the input
tensor lives on (CPU fallback included) and supports the three distances the
conversion templates / GUI expose: ``euclidean``, ``cosine`` and ``manhattan``.

Scalability
-----------
Exact Lloyd's k-means costs ``O(N*K*D)`` per iteration. The SOG-* templates ask
for a spherical-harmonics codebook of ``K = 65536`` clusters, which is far too
large to fit on the full primitive set. We therefore fit the centroids on a
bounded random subsample (reservoir-style: ``min(N, max(4*K, 200_000))`` points,
capped at ``max_samples``) and then run a single chunked full-data assignment
pass so that every primitive gets a label consistent with the returned
centroids. The ``N x K`` score matrix is never materialised in full: work is
done in row chunks sized to a fixed element budget.

The public surface intentionally mirrors the small part of torchpq's API that
:class:`ffsplat.models.transformations.Cluster` relies on::

    kmeans = KMeans(n_clusters=65536, distance="euclidean", verbose=True)
    labels = kmeans.fit(x)        # x: [D, N] -> labels: [N] int64, same device
    centroids = kmeans.centroids  # [D, K] float32, same device
"""

from __future__ import annotations

import torch
from torch import Tensor

DISTANCES = ("euclidean", "cosine", "manhattan")

# Upper bound on the number of score entries (rows x clusters) held at once.
_MAX_SCORE_ELEMS = 1 << 25


def _l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Row-wise L2 normalisation, with ``eps`` flooring the norm."""
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def _chunk_rows(n_clusters: int, max_rows: int = 8192) -> int:
    """Rows per chunk so that ``rows * n_clusters`` stays within the score budget."""
    return max(1, min(max_rows, _MAX_SCORE_ELEMS // max(n_clusters, 1)))


def _distances(rows: Tensor, centroids: Tensor, distance: str) -> Tensor:
    """``[C, D] x [K, D] -> [C, K]`` scores for one chunk of rows."""
    match distance:
        case "euclidean":
            # torch.cdist(p=2) is one fused matmul-based kernel; the hand-expanded
            # ``||x||^2 - 2 x.c + ||c||^2`` needed three extra passes over a [C, K]
            # buffer. Measured on the real SOG shapes (N=262144, K=65536, D=45) on an
            # M1 Max: 3.0x faster on MPS (5.4 s -> 1.95 s per full assignment pass) and
            # 1.2x faster on CPU, with identical labels (262144/262144).
            return torch.cdist(rows, centroids, p=2)
        case "cosine":
            return 1.0 - _l2_normalize(rows) @ _l2_normalize(centroids).t()
        case "manhattan":
            return _manhattan_blocks(rows, centroids)
    raise ValueError(f"Unknown distance metric: {distance!r} (expected one of {DISTANCES})")


def _manhattan_blocks(rows: Tensor, centroids: Tensor) -> Tensor:
    """``[C, D] x [K, D] -> [C, K]`` L1 scores, summed over centroid blocks.

    The L1 metric has no matmul identity, so the score matrix has to be materialised by
    subtraction; the only questions are how big a block to materialise and whether to use a
    fused kernel. ``torch.cdist(p=1)`` is the fused kernel, but on a whole row chunk it is
    unsafe on MPS: its implementation allocates an internal ``[C, K, D]`` buffer, and whether
    that turns out to be merely slow, rejected (``RuntimeError: Invalid buffer size``) or - the
    dangerous case - silently wrong depends on the buffer size *and* the allocator state, not on
    a clean row threshold. Observed on an M1 Max with K=40804, D=45: 4/40804 correct labels for
    C=1644 in one process, 6/40804 for the same C=1644 in a second process after a C=822 call,
    and ``Invalid buffer size: 22.50 GiB`` at C=3287 (22.49 GiB at C=3288 in a second, separate
    run). A codec that can mislabel a quarter of the primitives is worse than one that is 2x
    slower, so the block size is bounded by ``_MAX_SCORE_ELEMS`` and ``cdist`` is called per
    *centroid block* on every device. That keeps the fused kernel with a bounded peak allocation,
    and was measured on the real shapes (one full assignment pass, labels compared element-wise
    against the previous implementation, and re-checked at C=8192/B=400):

        N=40804,  K=40804,  D=45:  cpu 11.96 s ->  3.98 s   mps 5.00 s -> 4.12 s
        N=262144, K=65536, D=45:  cpu 118.19 s -> 38.24 s   mps 56.83 s -> 47.98 s
        labels identical in all four comparisons (40804/40804, 262144/262144).
    """
    block = max(1, _MAX_SCORE_ELEMS // max(rows.shape[0], 1) // max(rows.shape[1], 1))
    out = rows.new_empty((rows.shape[0], centroids.shape[0]))
    for start in range(0, centroids.shape[0], block):
        stop = min(start + block, centroids.shape[0])
        out[:, start:stop] = torch.cdist(rows, centroids[start:stop], p=1)
    return out


def _assign(x: Tensor, centroids: Tensor, distance: str, chunk: int) -> Tensor:
    """Chunked nearest-centroid assignment. ``x: [N, D]`` -> labels ``[N]`` int64."""
    labels = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        labels[start:stop] = _distances(x[start:stop], centroids, distance).argmin(dim=1)
    return labels


class KMeans:
    """Minimal drop-in replacement for ``torchpq.clustering.KMeans``.

    Args:
        n_clusters: Number of clusters (codebook entries) to fit.
        distance: One of ``"euclidean"``, ``"cosine"``, ``"manhattan"``.
        verbose: Print per-iteration progress.
        max_iter: Lloyd iterations on the fit set (early-stopped on label stability).
        seed: Seed for the subsample / initialisation.
        max_samples: Hard cap on the number of points used for the Lloyd search.
    """

    def __init__(
        self,
        n_clusters: int,
        distance: str = "euclidean",
        verbose: bool = False,
        max_iter: int = 10,
        seed: int = 42,
        max_samples: int = 1_000_000,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if distance not in DISTANCES:
            raise ValueError(f"Unknown distance metric: {distance!r} (expected one of {DISTANCES})")
        self.n_clusters = int(n_clusters)
        self.distance = distance
        self.verbose = verbose
        self.max_iter = max(1, int(max_iter))
        self.seed = int(seed)
        self.max_samples = int(max_samples)
        self._centroids: Tensor | None = None

    @property
    def centroids(self) -> Tensor:
        """``[D, K]`` float32 centroids on the device the input lived on."""
        if self._centroids is None:
            raise RuntimeError("KMeans.fit() must be called before accessing centroids")
        return self._centroids

    def fit(self, x: Tensor) -> Tensor:
        """Fit the codebook. ``x`` is ``[D, N]`` (features x samples, as torchpq used)."""
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D tensor [D, N], got shape {tuple(x.shape)}")
        n = x.shape[1]
        if n == 0:
            raise ValueError("Cannot fit k-means on an empty tensor")
        k = min(self.n_clusters, n)

        with torch.no_grad():
            samples = x.to(torch.float32).t().contiguous()  # [N, D]
            fit_data, generator = self._select_fit_set(samples, n)
            centroids = self._initialise(fit_data, k, generator)
            labels = self._lloyd(fit_data, centroids)
            # A single full-data pass so every sample gets a label consistent with
            # the centroids we return (fit_data may have been a subsample).
            if fit_data.shape[0] != n:
                labels = _assign(samples, centroids, self.distance, _chunk_rows(k))

            self._centroids = centroids.t().contiguous()  # [D, K]
            if self.verbose:
                _, counts = labels.unique(return_counts=True)
                print(f"k-means: done - {counts.numel()} non-empty clusters of {k}")
            return labels

    def _select_fit_set(self, samples: Tensor, n: int) -> tuple[Tensor, torch.Generator]:
        """Bounded random subsample used for the Lloyd search (``samples`` is ``[N, D]``)."""
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        cap = min(n, self.max_samples, max(4 * self.n_clusters, 200_000))
        if cap >= n:
            if self.verbose:
                print(f"k-means: fitting on all {n} samples ({self.n_clusters} clusters, {self.distance})")
            return samples, generator

        index = torch.randperm(n, generator=generator)[:cap].to(samples.device)
        if self.verbose:
            print(f"k-means: fitting on a {cap}/{n} subsample ({self.n_clusters} clusters, {self.distance})")
        return samples[index], generator

    def _initialise(self, fit_data: Tensor, k: int, generator: torch.Generator) -> Tensor:
        """Random distinct-point initialisation.

        k-means++ would need ``k`` passes over the data, which is not affordable at
        ``K = 65536`` for the SOG codebooks.
        """
        init_index = torch.randperm(fit_data.shape[0], generator=generator)[:k].to(fit_data.device)
        return fit_data[init_index].clone()

    def _lloyd(self, fit_data: Tensor, centroids: Tensor) -> Tensor:
        """Alternate assignment and exact centroid update until the labels settle."""
        chunk = _chunk_rows(centroids.shape[0])
        labels = None
        for iteration in range(self.max_iter):
            new_labels = _assign(fit_data, centroids, self.distance, chunk)
            if labels is not None and torch.equal(new_labels, labels):
                if self.verbose:
                    print(f"k-means: converged after {iteration} iteration(s)")
                return new_labels
            labels = new_labels
            centroids = self._update_centroids(fit_data, labels, centroids)
            if self.verbose:
                print(f"k-means: iteration {iteration + 1}/{self.max_iter} done")
        # The loop moves the centroids one step ahead of the labels it computed, so
        # assign once more to return labels that match the returned centroids.
        return _assign(fit_data, centroids, self.distance, chunk)

    def _update_centroids(self, data: Tensor, labels: Tensor, prev: Tensor) -> Tensor:
        """Exact centroid update; empty clusters keep their previous position."""
        sums = torch.zeros_like(prev)
        counts = torch.zeros(prev.shape[0], dtype=torch.float32, device=prev.device)
        step = 1 << 20
        for start in range(0, data.shape[0], step):
            stop = min(start + step, data.shape[0])
            chunk_labels = labels[start:stop]
            sums.index_add_(0, chunk_labels, data[start:stop])
            counts.index_add_(0, chunk_labels, torch.ones_like(chunk_labels, dtype=torch.float32))
        nonempty = counts > 0
        prev[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        return prev
