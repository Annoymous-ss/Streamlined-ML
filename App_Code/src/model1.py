# Importing libaries 
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st

# model function to be used in the web app 
def model(x_train, y_train, x_test, y_test, epochs, lr):
    lg = LinearReg(lr, epochs)
    lg.fit(x_train, y_train)
    prediction = lg.predict(x_test)
    st.write("Here's the Actual v/s Predicted plot")
    ploty(y_test, prediction)
    accuracy(y_test, prediction)
# MultiLinear Regression class
class LinearReg():
    # constructor 
    def __init__(self, learning_rate = 0.0001, iterations = 1000):
        self.lr = learning_rate 
        self.epochs = iterations 
        self.weights = None
        self.b = 0
        
    # fit method 
    def fit(self, x_train, y_train):
        self.rows, self.columns = x_train.shape
        self.weights = np.zeros(self.columns)
        # calling the helper function to update the weights 
        for i in range(self.epochs):
            self.update_weights(x_train, y_train)
        return self
        
    # helper method 
    def update_weights(self, x_train, y_train):
        y_pred = self.predict(x_train)
        # calculating the gradients 
        d_w = (-2 * x_train.T.dot(y_train - y_pred)) / self.rows
        d_b = (-2 * np.sum(y_train - y_pred)) / self.rows
        #  updating 
        self.weights -= self.lr * d_w
        self.b -= self.lr * d_b

    # predict function 
    def predict(self, x_test):
        return x_test.dot(self.weights) + self.b

# Plot function
def ploty(y_test, prediction):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, prediction, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], # X-axis value range 
             [y_test.min(), y_test.max()], # Y-axis value range
             linestyle = '--', color = 'red', lw = 2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    st.pyplot(plt)


# Accuracy function
def accuracy(y_test, prediction):
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    metrics = pd.DataFrame(
        {
            "Metrics" : ["MAE", "MSE", "RMSE", "R2_Score"],
            "Score" : [mae, mse, rmse, r2]
        }
    )
    # printing them 
    st.write("Accuracy Metrics of the Model\n", metrics)

code = '''
# MultiLinear Regression class : Gradient-Descent approach
class LinearReg():
    # constructor 
    def __init__(self, learning_rate = 0.0001, iterations = 1000):
        self.lr = learning_rate 
        self.epochs = iterations 
        self.weights = None
        self.b = 0
        
    # fit method 
    def fit(self, x_train, y_train):
        self.rows, self.columns = x_train.shape
        self.weights = np.zeros(self.columns)
        # calling the helper function to update the weights 
        for i in range(self.epochs):
            self.update_weights(x_train, y_train)
        return self
        
    # helper method 
    def update_weights(self, x_train, y_train):
        y_pred = self.predict(x_train)
        # calculating the gradients 
        d_w = (-2 * x_train.T.dot(y_train - y_pred)) / self.rows
        d_b = (-2 * np.sum(y_train - y_pred)) / self.rows
        #  updating 
        self.weights -= self.lr * d_w
        self.b -= self.lr * d_b

    # predict function 
    def predict(self, x_test):
        return x_test.dot(self.weights) + self.b
'''