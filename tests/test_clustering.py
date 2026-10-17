"""Tests for :mod:`ffsplat.models.clustering`.

This module replaced the CUDA-only ``torchpq.clustering.KMeans``. The tests cover the
properties the SOG codebook depends on: the documented ``[D, N]`` input convention,
recovery of well-separated structure, every advertised distance metric, the
``n_clusters > n_samples`` clamp and device-agnostic execution (MPS when available).
"""

from __future__ import annotations

import pytest
import torch

from ffsplat.models.clustering import DISTANCES, KMeans


def _three_wells(n_per_well: int = 300, dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[N, dim]`` samples from 3 well-separated wells plus their true labels."""
    centres = torch.tensor([[-10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(0)
    points, truth = [], []
    for label, centre in enumerate(centres):
        points.append(centre + 0.1 * torch.randn(n_per_well, dim, generator=generator))
        truth.append(torch.full((n_per_well,), label, dtype=torch.int64))
    return torch.cat(points), torch.cat(truth)


def _is_pure_clustering(labels: torch.Tensor, truth: torch.Tensor) -> bool:
    """True iff every cluster contains samples from exactly one ground-truth well.

    Cluster ids are arbitrary, so the recovered labelling is only ever checked up to a
    permutation of the ids.
    """
    return all(truth[labels == label].unique().numel() == 1 for label in labels.unique())


@pytest.mark.parametrize("distance", DISTANCES)
def test_kmeans_recovers_well_separated_clusters(distance: str) -> None:
    samples, truth = _three_wells()
    kmeans = KMeans(n_clusters=3, distance=distance, seed=0)
    labels = kmeans.fit(samples.t().contiguous())  # documented [D, N] convention

    assert labels.shape == (samples.shape[0],)
    assert labels.dtype == torch.int64
    assert kmeans.centroids.shape == (samples.shape[1], 3)
    assert kmeans.centroids.dtype == torch.float32
    assert labels.unique().numel() == 3, "did not produce 3 clusters"
    assert _is_pure_clustering(labels, truth), "a cluster mixed two wells"


def test_kmeans_centroids_land_on_the_true_well_centres() -> None:
    """For euclidean k-means the recovered centroids must sit on the wells."""
    samples, _ = _three_wells()
    kmeans = KMeans(n_clusters=3, distance="euclidean", seed=0)
    kmeans.fit(samples.t())
    centres = torch.tensor([[-10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    # permutation-invariant: every recovered centroid is close to some true centre
    dists = torch.cdist(kmeans.centroids.t(), centres)
    assert float(dists.min(dim=1).values.max()) < 0.5


def test_kmeans_deterministic_for_a_fixed_seed() -> None:
    samples, _ = _three_wells()
    a = KMeans(n_clusters=8, seed=7).fit(samples.t())
    b = KMeans(n_clusters=8, seed=7).fit(samples.t())
    assert torch.equal(a, b)


def test_kmeans_clamps_cluster_count_to_sample_count() -> None:
    samples = torch.randn(10, 5)
    kmeans = KMeans(n_clusters=64, seed=0)
    labels = kmeans.fit(samples.t())
    assert kmeans.centroids.shape == (5, 10)
    assert int(labels.max()) < 10


def test_kmeans_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=4, distance="chebyshev")
    with pytest.raises(ValueError):
        KMeans(n_clusters=4).fit(torch.randn(3))
    with pytest.raises(RuntimeError):
        _ = KMeans(n_clusters=4).centroids


def test_kmeans_matches_cpu_on_mps() -> None:
    """The MPS code path must agree with CPU on the same data."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    samples, truth = _three_wells()
    cpu_labels = KMeans(n_clusters=3, seed=0).fit(samples.t())
    mps_labels = KMeans(n_clusters=3, seed=0).fit(samples.t().to("mps"))
    assert torch.equal(mps_labels.cpu(), cpu_labels)
    assert _is_pure_clustering(cpu_labels, truth)


def test_kmeans_full_data_pass_labels_every_sample() -> None:
    """With more samples than ``max_samples`` every sample still gets a label."""
    samples = torch.randn(5000, 6)
    kmeans = KMeans(n_clusters=16, seed=1, max_samples=1000)
    labels = kmeans.fit(samples.t())
    assert labels.shape == (5000,)
    assert int(labels.min()) >= 0 and int(labels.max()) < 16
