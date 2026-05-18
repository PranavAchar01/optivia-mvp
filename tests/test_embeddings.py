"""Tests for the embeddings module (§4.5, §5.7)."""

import pytest

from backend.embeddings import cosine_similarity


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 0, 0], [1.0, 0, 0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0, 0], [0, 1.0, 0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0, 0], [-1.0, 0, 0]) == pytest.approx(-1.0)


def test_cosine_empty_vectors_safe():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], []) == 0.0


def test_cosine_zero_vector_safe():
    assert cosine_similarity([0, 0, 0], [1.0, 1.0, 1.0]) == 0.0


def test_cosine_mismatched_lengths_safe():
    assert cosine_similarity([1.0, 0], [1.0, 0, 0]) == 0.0
