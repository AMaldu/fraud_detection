import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from scipy.sparse import save_npz

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.head(50000)  # Remember to remove this line 
    print(f"Data loaded with {len(df)} rows")
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

def create_pipeline(preprocessor):
    # Define the SMOTE step
    smote = SMOTE(random_state=42)
    
    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote)
    ])
    return pipeline

def preprocess_and_balance_data(pipeline, X_train, y_train, X_test):
    # Apply preprocessing and SMOTE to the training data
    X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)
    # Apply preprocessing to the test data (SMOTE is not applied to the test data)
    X_test_scaled = pipeline.named_steps['preprocessor'].transform(X_test)
    return X_train_balanced, y_train_balanced, X_test_scaled

def save_data(X_train_balanced, X_test_scaled, y_train_balanced, y_test, save_dir):
    save_npz(os.path.join(save_dir, 'X_train_balanced.npz'), X_train_balanced)
    save_npz(os.path.join(save_dir, 'X_test_scaled.npz'), X_test_scaled)
    joblib.dump(y_train_balanced, os.path.join(save_dir, 'y_train_balanced.pkl'))
    joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))

def main():
    # Define file paths
    filepath = os.path.abspath('../fraud_detection/data/bronze/PS_20174392719_1491204439457_log.csv')
    save_dir = os.path.abspath('../fraud_detection/data/gold/')
    
    # Load and prepare data
    df = load_and_prepare_data(filepath)
    X, y = split_features_labels(df)
    
    # Define categorical and numeric features
    categorical_features = ['type', 'nameDest']
    numeric_features = ['step', 'amount', 'oldbalanceOrig', 'oldbalanceDest', 'diffbalanceOrig', 'diffbalanceDest']
    
    # Create preprocessor and pipeline
    preprocessor = create_preprocessor(categorical_features, numeric_features)
    pipeline = create_pipeline(preprocessor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess and balance data
    X_train_balanced, y_train_balanced, X_test_scaled = preprocess_and_balance_data(pipeline, X_train, y_train, X_test)
    
    # Save data
    save_data(X_train_balanced, X_test_scaled, y_train_balanced, y_test, save_dir)
    
    print("Preprocessing and balancing using Pipeline.")

if __name__ == "__main__":
    main()
