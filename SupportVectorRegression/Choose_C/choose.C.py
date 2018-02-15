print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold

ax = plt.subplot(111)
# Import data
data = np.genfromtxt(sys.argv[1],delimiter=None)
temp=data[:,0]
conc=data[:,1]
time=data[:,2]
dens=data[:,3]
tcts=data[:,[0,1,2]]
size=tcts.shape[0]*tcts.shape[1]
toln=data[:,0].shape[0]
print "toln is ",toln

# Define Three SVR algorithms
CA=[1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6]
lmean1=list()
lstdv1=list()
lmean2=list()
lstdv2=list()
lmean3=list()
lstdv3=list()
for C in CA: 
  print "C=", C
  svr_rbf = SVR(kernel='rbf', C=C, gamma=0.1)
  svr_lin = SVR(kernel='linear', C=C)
  svr_poly = SVR(kernel='poly', C=C, degree=6)
  
  # Generate KFold Cross-Validation data
  # Then use for loop to evalute each CV sample
  X = np.arange(toln)
  random_state=10001890
  rkf= RepeatedKFold(n_splits=4, n_repeats=6, random_state=random_state)
  lerr1=list()
  lerr2=list()
  lerr3=list()
  for train, test in rkf.split(X):
    X_train, X_test, y_train, y_test = tcts[train], tcts[test], dens[train], dens[test]
    tsize=y_test.shape[0]*1
    y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
    y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
    y_poly= svr_poly.fit(X_train, y_train).predict(X_test)
   
    err_rbf=np.linalg.norm(y_rbf-y_test)/np.sqrt(tsize)
    err_lin=np.linalg.norm(y_lin-y_test)/np.sqrt(tsize)
    err_poly=np.linalg.norm(y_poly-y_test)/np.sqrt(tsize)
    lerr1.append(err_rbf)
    lerr2.append(err_lin)
    lerr3.append(err_poly)
  
  # Calculate mean and stdev
  err1=np.array(lerr1)
  err2=np.array(lerr2)
  err3=np.array(lerr3)
  merr1=err1.mean()
  merr2=err2.mean()
  merr3=err3.mean()
  serr1=err1.std()
  serr2=err2.std()
  serr3=err3.std()
  lmean1.append(merr1)
  lmean2.append(merr2)
  lmean3.append(merr3)
  lstdv1.append(serr1)
  lstdv2.append(serr2)
  lstdv3.append(serr3)

mean1=np.array(lmean1)
stdv1=np.array(lstdv1)
mean2=np.array(lmean2)
stdv2=np.array(lstdv2)
mean3=np.array(lmean3)
stdv3=np.array(lstdv3)

ax.set_xscale("log")
plt.errorbar(CA, mean1, stdv1,color='navy',marker='s', label='RBF Kernel')
plt.errorbar(CA, mean2, stdv2,color='c',   marker='s', label='Linear Kernel')
plt.errorbar(CA, mean3, stdv3,color='red', marker='s', label='Polynomial Kernel')
plt.xlabel('C-value')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('choose.C.plot.png',dpi=300,format='png')
