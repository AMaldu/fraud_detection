import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def pie_plot(col):
    trans_freq = col.value_counts()
    explode = [0.02] * len(trans_freq)

    plt.figure(figsize=(5, 5))
    plt.pie(trans_freq, labels=trans_freq.index, autopct='%1.1f%%', startangle=0, colors=plt.cm.Set2.colors, explode=explode)
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