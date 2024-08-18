import unittest

import numpy as np
import pandas as pd

from src.utils import skewness_and_kurtosis


class TestVisualizations(unittest.TestCase):

    def test_skewness_and_kurtosis(self):
        df = pd.DataFrame({"values": np.random.normal(0, 1, 1000)})
        result = skewness_and_kurtosis(df, "values")
        self.assertIsInstance(result, dict)
        self.assertIn("skewness", result)
        self.assertIn("kurtosis", result)
        self.assertIsInstance(result["skewness"], float)
        self.assertIsInstance(result["kurtosis"], float)
