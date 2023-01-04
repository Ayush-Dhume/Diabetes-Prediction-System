from sklearn.preprocessing import StandardScaler
import pandas as pd


def return_scalar():
    df = pd.read_csv('diabetes.csv')

    X = df.drop(columns='Outcome', axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    print(standardized_data)
    return scaler

