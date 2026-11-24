import numpy as np

from data_contract import _reference_in_query_frame, stable_id


def test_stable_id_ignores_dictionary_order():
    assert stable_id("x", {"a": 1, "b": 2}) == stable_id("x", {"b": 2, "a": 1})


def test_reference_transform_uses_query_ego_frame():
    query = np.asarray([10.0, 20.0, np.pi / 2, 3.0, 0.0, 0.0, 0.0])
    reference = np.asarray([
        [1_000_000, 10.0, 20.0, np.pi / 2, 3.0],
        [1_100_000, 10.0, 21.0, np.pi / 2, 3.0],
    ])
    local = _reference_in_query_frame(reference, query, 1_000_000)
    assert np.allclose(local[:, 0], [0.0, 0.1])
    assert np.allclose(local[:, 1:3], [[0.0, 0.0], [1.0, 0.0]], atol=1e-12)
    assert np.allclose(local[:, 3], 0.0)
