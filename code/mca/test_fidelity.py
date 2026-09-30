"""Regression checks for self-exclusion and distance ties in geometric fidelity."""
import unittest
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from mca import (distance_average_ranks, distance_tie_blocks,
                 ordered_neighbors, neighborhood_fidelity)


class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(47).normal(size=(70, 7))
        self.keys = [('p01', f'e{i:03}') for i in range(len(self.x))]

    def orders(self, x):
        return ordered_neighbors(squareform(pdist(x)), self.keys)

    def test_duplicates_exclude_self(self):
        self.x[1:4] = self.x[0]
        orders = self.orders(self.x)
        for i, neighbors in enumerate(orders):
            self.assertNotIn(i, neighbors)
            self.assertEqual(len(set(neighbors)), len(self.x) - 1)
        np.testing.assert_array_equal(orders[0, :3], [1, 2, 3])
        np.testing.assert_array_equal(orders[2, :3], [0, 1, 3])

    def test_row_permutation(self):
        self.x[1:4] = self.x[0]
        order = np.random.default_rng(48).permutation(len(self.x))
        permuted = ordered_neighbors(squareform(pdist(self.x[order])),
                                     [self.keys[i] for i in order])
        np.testing.assert_array_equal(order[permuted], self.orders(self.x)[order])

    def test_complete_space_identity(self):
        full = self.orders(self.x)
        for k in (5, 10, 20):
            self.assertEqual(neighborhood_fidelity(full, full, k), (1., 1., 1.))

    def test_standard_trustworthiness_without_ties(self):
        for k in (5, 10, 20):
            actual = neighborhood_fidelity(self.orders(self.x),
                                           self.orders(self.x[:, :3]), k)[0]
            self.assertAlmostEqual(actual, trustworthiness(self.x, self.x[:, :3],
                                                          n_neighbors=k), places=14)

    def test_spearman_average_ranks(self):
        a = np.array([0., 0.25e-10, 0.75e-10, 2., 3.])
        np.testing.assert_array_equal(distance_average_ranks(a), [2., 2., 2., 4., 5.])
        a, b = pdist(self.x), pdist(self.x[:, :3])
        self.assertAlmostEqual(np.corrcoef(distance_average_ranks(a),
                                          distance_average_ranks(b))[0, 1],
                               spearmanr(a, b).statistic, places=14)

    def test_tie_blocks_do_not_chain(self):
        blocks = [v.tolist() for v in distance_tie_blocks(np.array([0., 0.75e-10, 1.5e-10]))]
        self.assertEqual(blocks, [[0, 1], [2]])


if __name__ == '__main__':
    unittest.main()
