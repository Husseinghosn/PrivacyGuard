from importlib.resources import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


training_url = "./data/adult-training.csv"
testing_url = "./data/adult-test.csv"

# merge training and testing datasets for EDA

training_df = pd.read_csv(training_url, header=None, skiprows=1 )
testing_df = pd.read_csv(testing_url, header=None, skiprows=1)
print(training_df.shape, testing_df.shape)
df = pd.concat([training_df, testing_df], ignore_index=True)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

df = pd.read_csv(url, header=None)
# Set plot_flag to True if you want to generate plots
plot_flag = False
def summarize_dataframe(df):
    print("--------------------------------------------------------------------")
    print("DataFrame Summary")
    print("--------------------------------------------------------------------")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}\n")

    print("The column names in the DataFrame are:\n", df.columns.tolist())
    print("\n")
    
    print("Column Data Types:")
    print(df.dtypes)
    print("\n")

    print("Missing Values per Column:")
    print(df.isnull().sum())
    print("\n")

    print("Statistical Summary of Numerical Columns:")
    print(df.describe().T)
    print("\n")

    print("Statistical Summary of Categorical Columns:")
    print(df.describe(include=["object"]).T)
    print("\n")

    print("Unique Values per Column:")
    for col in df.columns:
        unique_values = df[col].nunique()
        print(f"Column {col}: {unique_values} unique values")
    print("--------------------------------------------------------------------")

    print("\n")

    print("The number of duplicate rows in the DataFrame:\n", df.duplicated().sum())

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

    
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns


outlier_info = []
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    mask = (df[col] < lb) | (df[col] > ub)
    cnt = int(mask.sum())
    pct = cnt / len(df) * 100
    outlier_info.append((col, cnt, pct, lb, ub, mask))

outlier_df = pd.DataFrame([{"col": c, "count": cnt, "pct": pct} for c, cnt, pct, *_ in outlier_info]).sort_values("count", ascending=False)

plt.figure(figsize=(8, max(4, 0.5 * len(outlier_df))))
sns.barplot(x="count", y="col", data=outlier_df, palette="viridis")
plt.title("Outlier Counts per Numerical Column")
plt.xlabel("Outlier count")
plt.ylabel("Column")
plt.show()
