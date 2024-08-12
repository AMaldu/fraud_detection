import unittest
import os
import pandas as pd
import numpy as np
import pickle
from your_module import (
    pie_plot, scatter_plot, skewness_and_kurtosis, 
    detect_outliers, qq_plots, load_from_pickle, save_to_pickle
)
import matplotlib.pyplot as plt

class TestAnalysisFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configurar datos de prueba
        cls.df = pd.DataFrame({
            'isFraud': [0, 1, 0, 1, 0],
            'isflaggedfraud': [0, 0, 0, 0, 1],
            'amount': [100.0, 200.0, 150.0, 300.0, 50.0],
            'oldbalanceOrig': [1000.0, 2000.0, 1500.0, 3000.0, 500.0],
            'diffbalanceOrig': [900.0, 1800.0, 1350.0, 2700.0, 450.0]
        })
        cls.numeric_cols = ['amount', 'oldbalanceOrig', 'diffbalanceOrig']
        cls.pickle_file_path = './test_data.pkl'

    def test_pie_plot(self):
        with self.assertRaises(TypeError):
            pie_plot(self.df['isFraud'])

    def test_scatter_plot(self):
        with self.assertRaises(ValueError):
            scatter_plot(self.df['isFraud'], self.df['isflaggedfraud'])

    def test_skewness_and_kurtosis(self):
        result = skewness_and_kurtosis(self.df, 'amount')
        self.assertIn('skewness', result)
        self.assertIn('kurtosis', result)
        self.assertIsInstance(result['skewness'], float)
        self.assertIsInstance(result['kurtosis'], float)

    def test_detect_outliers(self):
        with self.assertLogs(level='INFO') as cm:
            detect_outliers(self.df)
        self.assertGreaterEqual(len(cm.output), 0)



    def test_save_to_pickle(self):
        save_to_pickle(self.df, self.pickle_file_path)
        self.assertTrue(os.path.exists(self.pickle_file_path))

    def test_load_from_pickle(self):
        save_to_pickle(self.df, self.pickle_file_path)
        loaded_data = load_from_pickle(self.pickle_file_path)
        pd.testing.assert_frame_equal(loaded_data, self.df)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.pickle_file_path):
            os.remove(cls.pickle_file_path)

if __name__ == '__main__':
    unittest.main()
