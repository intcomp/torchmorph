import unittest

import torch

import torchmorph


class Test1DEuclideanDistanceTransform(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def test_basic_features(self):
        """Test with 32 elements and 3 feature points"""
        print("\n=== Test 1: Basic Features (32 elements) ===")
        input_tensor = torch.zeros(32, dtype=torch.float32, device=self.device)
        input_tensor[0] = 1.0
        input_tensor[12] = 1.0
        input_tensor[31] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)

        # Check specific positions
        self._check_position(dist, indices, 0, 0, 0)
        self._check_position(dist, indices, 6, 0, 6)
        self._check_position(dist, indices, 12, 12, 0)
        self._check_position(dist, indices, 21, 12, 9)
        self._check_position(dist, indices, 31, 31, 0)

    def test_multiple_features(self):
        """Test with multiple feature points"""
        print("\n=== Test 2: Multiple Features ===")
        input_tensor = torch.zeros(32, dtype=torch.float32, device=self.device)
        features = [0, 3, 7, 10, 14, 21, 31]
        input_tensor[features] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)

        for pos in range(32):
            self._verify_nearest(pos, dist, indices, features)

    def test_batch_processing(self):
        """Test with 2D array (batch of 1D rows)"""
        print("\n=== Test 3: Batch Processing (4x32) ===")
        input_tensor = torch.zeros(4, 32, dtype=torch.float32, device=self.device)
        input_tensor[0, [0, 15, 31]] = 1.0
        input_tensor[1, [5, 10, 20]] = 1.0
        input_tensor[2, [8, 24]] = 1.0
        input_tensor[3, [16]] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)
        self.assertEqual(dist.shape, (4, 32))

        # Check row 0
        features_row0 = [0, 15, 31]
        for pos in range(32):
            self._verify_nearest(pos, dist[0], indices[0], features_row0)

    def test_boundary_conditions(self):
        """Test empty and full feature arrays"""
        print("\n=== Test 4: Boundary Conditions ===")
        # No features
        input_empty = torch.zeros(32, dtype=torch.float32, device=self.device)
        dist_empty, idx_empty = torchmorph.distance_transform(input_empty)
        self.assertTrue(torch.all(dist_empty > 1000))  # Should be large/inf
        self.assertTrue(torch.all(idx_empty == -1))

        # All features
        input_full = torch.ones(32, dtype=torch.float32, device=self.device)
        dist_full, idx_full = torchmorph.distance_transform(input_full)
        self.assertTrue(torch.all(dist_full == 0))
        expected_idx = torch.arange(32, device=self.device, dtype=torch.int32).unsqueeze(-1)
        self.assertTrue(torch.all(idx_full == expected_idx))

    def test_large_array(self):
        """Test large array to verify cross-tile propagation"""
        print("\n=== Test 5: Large Array (1024 elements) ===")
        input_tensor = torch.zeros(1024, dtype=torch.float32, device=self.device)
        features = [0, 512, 1023]
        input_tensor[features] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)

        test_positions = [0, 256, 512, 768, 1023]
        for pos in test_positions:
            self._verify_nearest(pos, dist, indices.squeeze(), features)

    def test_cross_tile_boundary(self):
        """Test propagation across tile boundaries"""
        print("\n=== Test 6: Cross-Tile Propagation ===")
        # 768 elements (3 tiles of 256)
        input_tensor = torch.zeros(768, dtype=torch.float32, device=self.device)
        features = [100, 600]
        input_tensor[features] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)

        # Check around boundaries (256, 512)
        test_positions = [250, 255, 256, 260, 350, 500, 510, 512, 520]
        for pos in test_positions:
            self._verify_nearest(pos, dist, indices.squeeze(), features)

    def test_large_2d_batch(self):
        """Test large 2D batch"""
        print("\n=== Test 7: Large 2D Batch ===")
        input_tensor = torch.zeros(3, 600, dtype=torch.float32, device=self.device)
        rows_features = {
            0: [0, 299, 599],
            1: [150, 450],
            2: [300],
        }

        for row, feats in rows_features.items():
            input_tensor[row, feats] = 1.0

        dist, indices = torchmorph.distance_transform(input_tensor)

        # Verify specific points with dynamic calculation
        test_cases = [
            (0, 150),
            (0, 450),
            (1, 300),
            (2, 100),
            (2, 500),
        ]

        for row, pos in test_cases:
            self._verify_nearest(pos, dist[row], indices[row].squeeze(), rows_features[row])

    def _check_position(self, dist, indices, pos, expected_idx, expected_dist):
        actual_dist = dist[pos].item()
        actual_idx = indices[pos].item() if indices.ndim == 1 else indices[pos, 0].item()

        self.assertAlmostEqual(actual_dist, float(expected_dist), places=1)
        self.assertEqual(actual_idx, expected_idx)

    def _verify_nearest(self, pos, dist, indices, features):
        actual_dist = dist[pos].item()
        nearest_idx = indices[pos].item() if indices.ndim == 1 else indices[pos].item()

        # Calculate ground truth dynamically
        true_dists = [abs(pos - f) for f in features]
        min_dist = min(true_dists)
        candidates = [f for f, d in zip(features, true_dists) if d == min_dist]

        self.assertAlmostEqual(
            actual_dist, float(min_dist), places=1, msg=f"Distance mismatch at {pos}"
        )
        self.assertIn(nearest_idx, candidates, msg=f"Nearest index mismatch at {pos}")


if __name__ == "__main__":
    unittest.main()
