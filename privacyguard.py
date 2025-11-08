import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math




url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

df = pd.read_csv(url, header=None)
# print(df.head())
plot_flag = True

def summarize_dataframe(df):
    print('--------------------------------------------------------------------')
    print('DataFrame Summary')
    print('--------------------------------------------------------------------')
    print(f'Total Rows: {df.shape[0]}')
    print(f'Total Columns: {df.shape[1]}\n')

    print('Column Data Types:')
    print(df.dtypes)
    print('\n')

    print('Missing Values per Column:')
    print(df.isnull().sum())
    print('\n')

    print('Statistical Summary of Numerical Columns:')
    print(df.describe().T)
    print('\n')

    print('Statistical Summary of Categorical Columns:')
    print(df.describe(include=['object']).T)
    print('\n')

    print('Unique Values per Column:')
    for col in df.columns:
        unique_values = df[col].nunique()
        print(f'Column {col}: {unique_values} unique values')
    print('--------------------------------------------------------------------')

    print('\n')

    print('The number of duplicate rows in the DataFrame:\n', df.duplicated().sum())

summarize_dataframe(df)

def plot_dataframe(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    nrows = 5
    ncols = 2

    print('--------------------------------------------------------------------')
    print('Distribution of the Numerical Columns')
    print('--------------------------------------------------------------------')
    plt.figure(figsize=(14, 5*nrows))
    
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}', fontsize=20)

        plt.xlabel(col, fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=1)
    plt.show()
    plt.close()

    print('--------------------------------------------------------------------')
    print('Boxplot of the Numerical Columns')
    print('--------------------------------------------------------------------')
    plt.figure(figsize=(28, 48))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}', fontsize=20)
        plt.xlabel(col, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=1)
    plt.show()
    plt.close()

    print('--------------------------------------------------------------------')
    print('Correlation Matrix of Numerical Columns')
    print('--------------------------------------------------------------------')
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    plt.close()

    print('--------------------------------------------------------------------')
    print('Distribution of the Categorical Columns')
    print('--------------------------------------------------------------------')
    for i in range(0, len(categorical_cols), 6):
        subset = categorical_cols[i:i+6]
        nrows = math.ceil(len(subset) / ncols)

        plt.figure(figsize=(14, 5 * nrows))
        plt.suptitle("Categorical Column Distributions", fontsize=20, y=5)

        for j, col in enumerate(subset, 1):
            if col != 13:
                plt.subplot(nrows, ncols, j)
                sns.countplot(y=df[col])  # Flip axes for readability
                plt.title(f'Distribution of {col}', fontsize=16)
                plt.xlabel('Frequency', fontsize=14)
                plt.ylabel(col, fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.4)
        plt.show()
        plt.close()
    
    plt.figure(figsize=(14, 25))
    sns.countplot(y=df[13])
    plt.title('Distribution of 13', fontsize=16) 
    plt.xlabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

if plot_flag:
    plot_dataframe(df)

    






