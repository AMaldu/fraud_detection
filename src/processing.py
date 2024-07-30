



df_imputed_outliers = data.copy()
for column in columns_to_check:
    median = data[column].median()
    df_imputed_outliers.loc[outliers_indices[column], column] = median