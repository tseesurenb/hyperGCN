import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from similarity_func import cosine_similarity_by_top_k_new as cosine_similarity_by_top_k

# Assuming your cosine_similarity_by_top_k function is already defined

class TestCosineSimilarityByTopK(unittest.TestCase):
    def test_small_matrix(self):
        # Small test matrix
        matrix = np.array([[1, 0, 0],
                           [0, 1, 1],
                           [1, 1, 0]])

        # Expected output for top_k=1
        expected_output_top_1 = np.array([[0., 0., 0.70710678],
                                          [0., 0., 0.70710678],
                                          [0.70710678, 0., 0.]])

        # Expected output for top_k=2
        expected_output_top_2 = np.array([[0., 0., 0.70710678],
                                          [0., 0.70710678, 0.70710678],
                                          [0.70710678, 0.70710678, 0.]])

        # Testing top_k=1
        result_top_1 = cosine_similarity_by_top_k(matrix, top_k=1, self_sim=False, verbose=0)
        np.testing.assert_almost_equal(result_top_1, expected_output_top_1, decimal=6)

        # Testing top_k=2
        result_top_2 = cosine_similarity_by_top_k(matrix, top_k=2, self_sim=False, verbose=0)
        np.testing.assert_almost_equal(result_top_2, expected_output_top_2, decimal=6)

    def test_self_similarity(self):
        # Small test matrix
        matrix = np.array([[1, 0, 0],
                           [0, 1, 1],
                           [1, 1, 0]])

        # Expected output for top_k=2 with self-similarity
        expected_output_with_self = np.array([[1., 0., 0.70710678],
                                              [0., 1., 0.70710678],
                                              [0.70710678, 0.70710678, 1.]])

        # Testing with self-similarity
        result_with_self = cosine_similarity_by_top_k(matrix, top_k=2, self_sim=True, verbose=0)
        np.testing.assert_almost_equal(result_with_self, expected_output_with_self, decimal=6)

    def test_verbose_output(self):
        # Check if verbose mode works without errors
        matrix = np.array([[1, 0, 0],
                           [0, 1, 1],
                           [1, 1, 0]])
        # Just to ensure it runs without issues in verbose mode
        cosine_similarity_by_top_k(matrix, top_k=2, self_sim=False, verbose=1)

if __name__ == '__main__':
    unittest.main()
