import numpy as np
from sklearn.datasets import load_boston
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.linear_model

class LinearRegression():
    def __init__(self, lr = 0.0001, epoches = 100, epsilon = 0.001, batch_size = 10):
        self.learning_rate = lr
        self.epoches = epoches
        self.epsilon = epsilon
        self.batch_size = 5


    def fit(self, data, label):
        self.w =  np.random.randn(data.shape[1],1)
        for i in range(self.epoches):
            index = np.random.permutation(data.shape[0])
            X = data[index]
            y = label[index]
            for j in range(0, data.shape[0], self.batch_size):
                X_i = X[i:i+self.batch_size, :]
                y_i = y[i:i+self.batch_size]
                prediction = np.dot(X_i,self.w)
                self.w = self.w - (1/data.shape[0])*self.learning_rate*( X_i.T.dot(prediction - y_i))


    def predict(self, data):
        return np.dot(data,self.w)


if __name__ == "__main__":
    data = load_boston()
    
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                        test_size=0.33, random_state=42)
    
    y_train = np.expand_dims(y_train, axis = 1)
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(mean_squared_error(y_pred, y_test))

    reg = sklearn.linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print(mean_squared_error(y_pred, y_test))