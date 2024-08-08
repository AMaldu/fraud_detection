
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import save_npz
import os

# Cargar el DataFrame
df = pd.read_csv('../data/bronze/PS_20174392719_1491204439457_log.csv')
print('the original df is:')
print(df.head)

# Renombrar columnas y reemplazar valores
df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
#df['isFraud'] = df['isFraud'].replace({0: 'no_fraud', 1: 'fraud'})

# Crear nuevas características
df['diffbalanceOrig'] = df['newbalanceOrig'] - df['oldbalanceOrig']
df['diffbalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
print('the df ready with all the necessary columns is')
print(df.head())

# Eliminar columnas no deseadas
df = df.drop(columns=['newbalanceOrig', 'nameOrig', 'newbalanceDest', 'isFlaggedFraud'])

# Dividir en características (X) y etiquetas (y)
X = df.drop(columns='isFraud')
y = df['isFraud']
print(f'X has this shape: {X.head()}')
print(f'y has this shape: {y.head()}')

# Preprocesador para datos categóricos
categorical_features = ['type', 'nameDest']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocesador para datos numéricos
numeric_features = ['step', 'amount', 'oldbalanceOrig', 'oldbalanceDest', 'diffbalanceOrig', 'diffbalanceDest']
numeric_transformer = StandardScaler()

# Combinar los preprocesadores en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Crear el pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Dividir el DataFrame en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar el pipeline a los datos de entrenamiento
X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)


print(f'X_train_scaled is: {X_train_scaled}')
print(f'X_test_scaled is: {X_test_scaled}')
print(f'y_train is: {y_train}')
print(f'y_test is: {y_test}')


save_dir = '../data/gold/'

# Guardar matrices dispersas
save_npz(os.path.join(save_dir, 'X_train_scaled.npz'), X_train_scaled)
save_npz(os.path.join(save_dir, 'X_test_scaled.npz'), X_test_scaled)

# Guardar etiquetas
joblib.dump(y_train, os.path.join(save_dir, 'y_train.pkl'))
joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))










# pd.DataFrame(X_train_scaled).to_csv('data/gold/X_train_scaled.csv', index=False, header=False)
# pd.DataFrame(X_test_scaled).to_csv('data/gold/X_test_scaled.csv', index=False, header=False)

# # Guardar y_train e y_test como CSV
# y_train.to_frame().reset_index(drop=True).to_csv('data/gold/y_train.csv', index=False, header=False)
# y_test.to_frame().reset_index(drop=True).to_csv('data/gold/y_test.csv', index=False, header=False)


print("Preprocesamiento completado usando Pipeline.")