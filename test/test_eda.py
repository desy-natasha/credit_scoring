import unittest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO
import sys
import os

matplotlib.use('Agg')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import plot_custom, barplot_compare, mannwhitney_test, chi2_test

class TestCustomFunctions(unittest.TestCase):
    
    def create_mock_data(self):
        """Create mock data for testing."""
        self.mock_data = {
            'Label': [0, 1] * 50,
            'Credit amount': np.random.randint(100, 1000, 100),
            'Variable': np.random.randint(10, 50, 100)
        }
        self.df = pd.DataFrame(self.mock_data)
        
    def test_plot_custom(self):
        """Test the plot_custom method."""
        self.create_mock_data()
        fig = plt.figure()
        plot_custom(self.df, 'Variable', types="hist") 

        self.assertIsInstance(fig, plt.Figure)

    def test_barplot_compare(self):
        """Test the barplot_compare method."""
        self.create_mock_data()
        fig = plt.figure()
        barplot_compare(self.df, 'Variable')
        
        self.assertIsInstance(fig, plt.Figure)

    def test_mannwhitney_test(self):
        """Test the mannwhitney_test method."""
        self.create_mock_data()
        captured_output = StringIO()
        sys.stdout = captured_output
        
        mannwhitney_test(self.df, 'Label', 'Variable', 'Test Mann-Whitney')
        
        sys.stdout = sys.__stdout__
        self.assertIn('Non-Parametric t-test', captured_output.getvalue())
        self.assertIn('reject H0', captured_output.getvalue())

    def test_chi2_test(self):
        """Test the chi2_test method."""
        self.create_mock_data()
        captured_output = StringIO()
        sys.stdout = captured_output

        chi2_test(self.df, 'Variable')
        
        sys.stdout = sys.__stdout__
        self.assertIn('Chi-Square test', captured_output.getvalue())
        
if __name__ == '__main__':
    unittest.main()
