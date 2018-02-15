# onr2017
Scripts for machine learning at ONR project (2017-)

Prerequisites:
Python 2, Numpy, Sklearn and Matplotlib Packages

Data:
unnorma.input.data contains four columns, which are Temperature, Concentration, Dwelling Time and %Mass Change. 
norm.input.data contains four columns after normalization, consistent with unnorma.input.data. 

Machine Learning:
Four machine learning models are explored based on 19 bulk sample results. The target property is %Mass Change and they're normalized before the machine learning process. 
1. Multivariate Linear Regression 
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
2. Support Vector Regression
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
3. Kernel Ridge Regression
http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
4. Neural Network
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
