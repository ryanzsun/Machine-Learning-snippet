import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def gini_index(data):
    pass


if __name__ == "__main__":
    data = load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.33, random_state=42)

    classifier = CART()
    classifier.fit(X_train, y_train)
    