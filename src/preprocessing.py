import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import save_npz
import os

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
    df['diffbalanceOrig'] = df['newbalanceOrig'] - df['oldbalanceOrig']
    df['diffbalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df = df.drop(columns=['newbalanceOrig', 'nameOrig', 'newbalanceDest', 'isFlaggedFraud'])
    return df

def split_features_labels(df):
    X = df.drop(columns='isFraud')
    y = df['isFraud']
    return X, y

def create_preprocessor(categorical_features, numeric_features):
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def preprocess_data(preprocessor, X_train, X_test):
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    return X_train_scaled, X_test_scaled

def save_data(X_train_scaled, X_test_scaled, y_train, y_test, save_dir):
    save_npz(os.path.join(save_dir, 'X_train_scaled.npz'), X_train_scaled)
    save_npz(os.path.join(save_dir, 'X_test_scaled.npz'), X_test_scaled)
    joblib.dump(y_train, os.path.join(save_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))

def main():
    filepath = '../data/bronze/PS_20174392719_1491204439457_log.csv'
    save_dir = '../data/gold/'
    
    df = load_and_prepare_data(filepath)
    X, y = split_features_labels(df)
    
    categorical_features = ['type', 'nameDest']
    numeric_features = ['step', 'amount', 'oldbalanceOrig', 'oldbalanceDest', 'diffbalanceOrig', 'diffbalanceDest']
    
    preprocessor = create_preprocessor(categorical_features, numeric_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, X_test_scaled = preprocess_data(preprocessor, X_train, X_test)
    
    save_data(X_train_scaled, X_test_scaled, y_train, y_test, save_dir)
    
    print("Preprocess using Pipeline.")


