import os
import unittest

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from your_module import (
    create_preprocessor,
    load_and_prepare_data,
    preprocess_data,
    save_data,
    split_features_labels,
)


class TestFullPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.filepath = "./test_data/PS_test_log.csv"
        cls.save_dir = "./test_output"
        os.makedirs(cls.save_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "step": [1, 2],
                "type": ["PAYMENT", "TRANSFER"],
                "amount": [19839.639648, 1864.280029],
                "oldbalanceOrig": [170136.0, 21249.0],
                "newbalanceOrig": [19384.720703, 0.000000],
                "nameOrig": ["C1231006815", "C1666544295"],
                "oldbalanceDest": [0.0, 0.0],
                "newbalanceDest": [0.0, 0.0],
                "nameDest": ["M1979787155", "M2044282225"],
                "isFraud": [0, 1],
                "isFlaggedFraud": [0, 0],
            }
        )

        df.to_csv(cls.filepath, index=False)

    def test_full_pipeline(self):
        df = load_and_prepare_data(self.filepath)
        X, y = split_features_labels(df)

        categorical_features = ["type", "nameDest"]
        numeric_features = [
            "step",
            "amount",
            "oldbalanceOrig",
            "oldbalanceDest",
            "diffbalanceOrig",
            "diffbalanceDest",
        ]

        preprocessor = create_preprocessor(categorical_features, numeric_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled, X_test_scaled = preprocess_data(preprocessor, X_train, X_test)

        save_data(X_train_scaled, X_test_scaled, y_train, y_test, self.save_dir)

        self.assertTrue(
            os.path.exists(os.path.join(self.save_dir, "X_train_scaled.npz"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.save_dir, "X_test_scaled.npz"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.save_dir, "y_train.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.save_dir, "y_test.pkl")))

        X_train_loaded = load_npz(os.path.join(self.save_dir, "X_train_scaled.npz"))
        X_test_loaded = load_npz(os.path.join(self.save_dir, "X_test_scaled.npz"))
        y_train_loaded = joblib.load(os.path.join(self.save_dir, "y_train.pkl"))
        y_test_loaded = joblib.load(os.path.join(self.save_dir, "y_test.pkl"))

        self.assertEqual(X_train_scaled.shape, X_train_loaded.shape)
        self.assertEqual(X_test_scaled.shape, X_test_loaded.shape)
        self.assertTrue((y_train == y_train_loaded).all())
        self.assertTrue((y_test == y_test_loaded).all())

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)
        if os.path.exists(cls.save_dir):
            for file in os.listdir(cls.save_dir):
                os.remove(os.path.join(cls.save_dir, file))
            os.rmdir(cls.save_dir)


if __name__ == "__main__":
    unittest.main()
