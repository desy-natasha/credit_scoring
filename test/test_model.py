import unittest
import numpy as np
import pandas as pd
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import sys
import os

matplotlib.use('Agg')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import model, random_search

class TestModelFunctions(unittest.TestCase):

    def create_mock_data(self):
        """Create mock data for testing."""
        self.mock_data = {
            'Label': [0, 1] * 50,
            'Credit amount': np.random.randint(100, 1000, 100),
            'Variable': np.random.randint(10, 50, 100)
        }
        self.df = pd.DataFrame(self.mock_data)
     
        # split into train and test sets
        self.data_train, self.data_test = train_test_split(self.df, test_size=0.2,random_state=1,stratify=self.df['Label'])
        
        # splot features and labels
        self.X_train, self.y_train = self.data_train.drop(columns='Label',axis=1), self.data_train[['Label']]
        self.X_test, self.y_test = self.data_test.drop(columns='Label',axis=1), self.data_test[['Label']]

        # initialize a model
        self.model = RandomForestClassifier(random_state=0)

    def test_model_function(self):
        """Test the model method."""
        self.create_mock_data()
       
        trained_model, accuracy, precision, recall, f1, auc = model(
            self.model, self.X_train, self.X_test, self.y_train, self.y_test, pprint=False
        )
        
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)
    
    def test_random_search_function(self):
        """Test the random_search method."""
        self.create_mock_data()

        # define hyperparameter grid
        parameters = {
            'n_estimators': [10, 20],
            'max_depth': [2, 4],
            'min_samples_split': [2, 3]
        }

        best_model = random_search(parameters, self.X_train, self.y_train, self.model, iter=5, cv_default=True)

        self.assertIsInstance(best_model, RandomForestClassifier)

    def test_random_search_with_custom_cv(self):
        """Test the random_search method with the RepeatedStratifiedKFold."""
        self.create_mock_data()
        
        # define hyperparameter grid
        parameters = {
            'n_estimators': [5, 10],
            'max_depth': [2, 3]
        }

        best_model = random_search(parameters, self.X_train, self.y_train, self.model, iter=3, cv_default=False)

        self.assertIsInstance(best_model, RandomForestClassifier)

if __name__ == '__main__':
    unittest.main()
