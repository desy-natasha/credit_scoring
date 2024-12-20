import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import data_preparation, smote_tomek

class TestDataPreparation(unittest.TestCase):

    def create_mock_data(self):
        """Create mock data for testing."""
        self.mock_data = pd.DataFrame({
            'Label': [0, 1] * 50,
            'Credit amount': np.random.randint(100, 1000, 100),
            'Variable': np.random.randint(10, 50, 100)
        })
        self.prep = data_preparation()

    def test_sample_data(self):
        """Test the sample_data method."""
        self.create_mock_data()
        self.prep.sample_data(self.mock_data, pprint=False)
        self.assertEqual(self.prep.data_train.shape[1], self.mock_data.shape[1])
        self.assertAlmostEqual(len(self.prep.data_train) / len(self.mock_data), 0.78, delta=0.1)

    def test_fe_data(self):
        """Test the fe_data method."""
        self.create_mock_data()
        self.prep.sample_data(self.mock_data, pprint=False)
        self.prep.fe_data(scaling=True, pprint=False)

        self.assertIn('Log_Credit amount', self.prep.X_train.columns)
        self.assertIn('Log_Variable', self.prep.X_train.columns)
        self.assertTrue(np.issubdtype(self.prep.X_train['Log_Credit amount'].dtype, np.floating))

if __name__ == "__main__":
    unittest.main()
