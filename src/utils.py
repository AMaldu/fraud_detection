import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import probplot, skew, kurtosis

def pie_plot(col):
    trans_freq = col.value_counts()
    explode = [0.02] * len(trans_freq)

    plt.figure(figsize=(5, 5))
    plt.pie(trans_freq, 
            labels=trans_freq.index, 
            autopct='%1.1f%%', 
            startangle=0, 
            colors=plt.cm.Set2.colors, 
            explode=explode)
    plt.title(f'Percentatge of {col.name} values')
    plt.show()
    
    
    

def scatter_plot(col1, col2):
    plt.figure(figsize=(8, 6))
    plt.scatter(col1, col2, c=col1, cmap='viridis')
    plt.xlabel('isFraud')
    plt.ylabel('isflaggedfraud')
    plt.title("Matriz de Gráficos de Puntos para las Columnas 'fraud' e 'isflaggedfraud'")

    plt.colorbar(label='isFraud')
    plt.show()
    
    
    
    
def skewness_and_kurtosis(df, column_name):
    column_data = df[column_name]
    skewness_value = skew(column_data)
    kurtosis_value = kurtosis(column_data)
    
    return {
        'skewness': skewness_value,
        'kurtosis': kurtosis_value
    }
    
    
    
def detect_outliers(data):
    numeric_columns = data.select_dtypes(include=['int64', 'float64'])

    non_binary_columns = numeric_columns.loc[:, numeric_columns.nunique() > 2]

    Q1 = non_binary_columns.quantile(0.25)
    Q3 = non_binary_columns.quantile(0.75)
    IQR = Q3 - Q1

    def find_outliers(column):
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        if outliers.empty:
            return None, 0.0, None
        else:
            percentage = (len(outliers) / len(data)) * 100
            count_outliers = len(outliers)
            return column, percentage, count_outliers

    columns_with_outliers = [find_outliers(column) for column in non_binary_columns.columns]
    columns_with_outliers = [(column, percentage, count_outliers) for column, percentage, count_outliers in columns_with_outliers if column is not None]

    for column, percentage, count_outliers in columns_with_outliers:
        print(f"Column: {column}, Percentage of outliers: {percentage:.2f}%, Total number of outliers: {count_outliers}")



    
def qq_plots(data, numeric_cols):
    for column in numeric_cols:
        plt.figure(figsize=(8, 5))
        probplot(data[column], dist="norm", plot=plt)
        plt.title(f'Q-Q plot for {column}')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.grid(True)
        plt.show()