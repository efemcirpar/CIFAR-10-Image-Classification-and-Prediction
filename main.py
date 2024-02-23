import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import layers
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.optimizers import RMSprop


#Loading Dataset

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.reshape(50000,3072)
X_test = X_test.reshape(10000,3072)

#KNN

#Defining KNN and Its Parameter Grid for Hyperparameter Tuning
knn = KNeighborsClassifier()
knn_hptuning_grid = {'n_neighbors': [1, 3, 5], 'weights': ['uniform', 'distance']}

#Performing Hyperparameter and Finding The Best Parameters
knn_gridsearchcv = GridSearchCV(knn, knn_hptuning_grid, cv=5)
knn_gridsearchcv.fit(X_train, Y_train)

#Fittinng Best Hyperparameter
print("Best Parameters: ", knn_gridsearchcv.best_params_)
print("Best Score: ", knn_gridsearchcv.best_score_)

#Predicting With Hyperparameter
Y_pred = knn_gridsearchcv.predict(X_test)

#Finding Accuracy Rate
knn_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Set Accuracy: ", knn_accuracy) 

#Plotting Decision Boundary
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, Y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c= Y_pred, cmap='coolwarm')
plt.title("KNN with n_neighbors=5 and weight=distance")
plt.show()

#--------------------------------------------------------------------------------------------------------------
#Linear Regression

#Defining Linear Regression and Its Parameter Grid for Hyperparameter Tuning
lr = LinearRegression()
lr_hptuning_grid = {'fit_intercept': [True, False]}

#Performing Hyperparameter and Finding The Best Parameters
lr_gridsearchcv = GridSearchCV(lr, lr_hptuning_grid, cv=5, n_jobs=-1)
lr_gridsearchcv.fit(X_train, Y_train)

#Fittinng Best Hyperparameter
print("Best Hyperparameters:", lr_gridsearchcv.best_params_)
lr_best_model = lr_gridsearchcv.best_estimator_
lr_best_model.fit(X_train, Y_train)

#Predicting With Hyperparameter
Y_pred = lr_best_model.predict(X_test)

#Calculating Mean Squared Error Since Linear Regression Can't Give Accuracy Rate
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

#Plotting Decision Boundary
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, Y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c= Y_pred, cmap='coolwarm')
plt.title("LR with fit_intercept = True")
plt.show()

#--------------------------------------------------------------------------------------------------------------
#SVM

#Scale The Data with MinMaxScaler or StandardScaler
scaler = MinMaxScaler()

#scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Using Some Part of The Data Because My Computer Doesn't Allow To Run All
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test)

X_train = X_train.iloc[0:2000,:]
Y_train = Y_train.iloc[0:2000,:]
X_test = X_test.iloc[0:400,:]
Y_test = Y_test.iloc[0:400,:]

#Defining SVM and Its Parameter Grid for Hyperparameter Tuning
svm = SVC()
svm_hptuning_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

#Performing Hyperparameter and Finding The Best Parameters
svm_gridsearchcv = GridSearchCV(svm, svm_hptuning_grid, cv=5)
svm_gridsearchcv.fit(X_train, Y_train)
print("Best Hyperparameters:", svm_gridsearchcv.best_params_)

#Fitting Best Hyperparameter
svm = svm_gridsearchcv.best_estimator_
svm.fit(X_train, Y_train)

#Predicting With Hyperparameter
Y_pred = svm.predict(X_test)

#Finding Accuracy Rate
svm_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Set Accuracy: ", svm_accuracy) 

#Plotting Decision Boundary
svm = SVC(C = 10 , kernel = 'rbf')
svm.fit(X_train, Y_train)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c= Y_pred, s=30, cmap=plt.cm.Paired)
plt.title("SVM with C=10 and Kernel = 'rbf'")
plt.show()

#--------------------------------------------------------------------------------------------------------------
#MLP

#Defining MLP and Its Parameter Grid for Hyperparameter Tuning
mlp = MLPClassifier(max_iter=100, random_state=1)
mlp_hptuning_grid = {'hidden_layer_sizes': [(5,), (10,)], 'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive'],}

#Performing Hyperparameter and Finding The Best Parameters
mlp_gridsearchcv = GridSearchCV(mlp, mlp_hptuning_grid, cv=5)
mlp_gridsearchcv.fit(X_train, Y_train)
print("Best Hyperparameters:", mlp_gridsearchcv.best_params_)

#Fittinng Best Hyperparameter
mlp = mlp_gridsearchcv.best_estimator_
mlp.fit(X_train, Y_train)

#Predicting With Hyperparameter
Y_pred = mlp.predict(X_test)

#Finding Accuracy Rate
mlp_accuracy = accuracy_score(Y_test, Y_pred)
print("Test Set Accuracy: ", mlp_accuracy) 

#Plotting Decision Boundary
mlp = MLPClassifier(hidden_layer_sizes = 5, alpha = 0.0001, learning_rate = 'constant' )
mlp.fit(X_train, Y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c= Y_pred, cmap='coolwarm')
plt.title("MLP with hidden_layer_size = (5,0), alpha = 0.0001, learning_rate = 'constant'")
plt.show()

#--------------------------------------------------------------------------------------------------------------
#CNN

#Getting Data Again Because We Should Not Reshape In CNN
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#Normalizing Data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Converts Labels To Categorical Class Amount = 10
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

#CNN Architecture
cnn = Sequential()

cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

#Compiling CNN
cnn.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, decay=1e-6), metrics=['accuracy'])

#Training CNN
history = cnn.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

#Evaluating CNN
score = cnn.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Plot The Accuracy and Loss According to Epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()