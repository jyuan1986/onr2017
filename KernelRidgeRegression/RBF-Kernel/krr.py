print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import sklearn
from sklearn.kernel_ridge import KernelRidge 
from sklearn.model_selection import RepeatedKFold
from scipy.stats import pearsonr

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

# Define KRR algorithms
#alpha=1/(2*C)
alpha=1e-4
krr_rbf = KernelRidge(kernel='rbf', alpha=alpha, gamma=0.1)

# Generate KFold Cross-Validation data
# Then use for loop to evalute each CV sample
X = np.arange(toln)
random_state=10001890
rkf= RepeatedKFold(n_splits=4, n_repeats=6, random_state=random_state)
lden1=list()
lden2=list()
lden3=list()
lerr1=list()
lerr2=list()
lerr3=list()
lerr4=list()

for train, test in rkf.split(X):
  X_train, X_test, y_train, y_test = tcts[train], tcts[test], dens[train], dens[test]
  trsize=y_train.shape[0]*1
  ttsize=y_test.shape[0]*1
  ytr_rbf = krr_rbf.fit(X_train, y_train).predict(X_train)
  ytt_rbf = krr_rbf.fit(X_train, y_train).predict(X_test)
  y_rbf   = krr_rbf.fit(X_train, y_train).predict(tcts)
  lden1.append(y_rbf) 
  err1_rbf=np.linalg.norm(ytr_rbf-y_train)/np.sqrt(trsize)
  err2_rbf=np.linalg.norm(ytt_rbf-y_test)/np.sqrt(ttsize)
  pcc1_rbf=pearsonr(ytr_rbf,y_train)[0]
  pcc2_rbf=pearsonr(ytt_rbf,y_test)[0]
  lerr1.append(err1_rbf)
  lerr2.append(err2_rbf)
  lerr3.append(pcc1_rbf)
  lerr4.append(pcc2_rbf)

# Calculate mean and stdev
den1=np.array(lden1)
mden1=np.mean(den1,axis=0)
sden1=np.std(den1,axis=0)

# Calculate mean and stdev
err1=np.array(lerr1)
err2=np.array(lerr2)
err3=np.array(lerr3)
err4=np.array(lerr4)
print "mden1=",mden1
print "shape(mden1)=",mden1.shape
print "shape(send1)=",sden1.shape
print "shape(dens)=",dens.shape
print "mean(rmse-train)=",err1.mean(),"stdev(rmse-train)",err1.std()
print "mean(rmse-test)=",err2.mean(), "stdev(rmse-test)",err2.std()
print "mean(pcc-train)=",err3.mean(), "stdev(pcc-train)",err3.std()
print "mean(pcc-test)=",err4.mean(),  "stdev(pcc-test)",err4.std()

lw = 2
plt.plot(dens,dens, color='black',lw=lw, label='data')
plt.errorbar(dens, mden1, sden1,color='navy',marker='s', label='KRR model',linestyle='None')
plt.xlabel('%Mass Change (experiment)')
plt.ylabel('%Mass Change (KRR Prediction)')
plt.title('Kernel Ridge Regression')
plt.legend()
plt.savefig('exp.vs.ml.plot.png',dpi=300,format='png')
