print(__doc__)

import numpy as np
import sys as sys
import sklearn
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

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

# Generate KFold Cross-Validation data
# Then use for loop to evalute each CV sample
X = np.arange(toln)
random_state=10001
alpha=1e-5
##MA,NA are dummy parameters##
MA=[1]
NA=[1]
ltden=list()
lsden=list()
lterr=list()
lserr=list()
lterrt=list()
lserrt=list()
ltpcc=list()
lspcc=list()
ltpcct=list()
lspcct=list()
lind1=list()
lind2=list()
for m in MA:
 for n in NA:
  rkf= RepeatedKFold(n_splits=4, n_repeats=6, random_state=random_state)
  clf= LinearRegression()
  
  lden=list() 
  lerr=list()
  lerrt=list()
  lpcc=list()
  lpcct=list()
  for train, test in rkf.split(X):
   X_train, X_test, y_train, y_test = tcts[train], tcts[test], dens[train], dens[test]
   tsize=y_test.shape[0]*1
   clf.fit(X_train, y_train)
   yt_nn=clf.predict(X_train)
   errt_nn=np.linalg.norm(yt_nn-y_train)/np.sqrt(y_train.shape[0])
   lerrt.append(errt_nn) ##root mean square error on train data
   y_nn=clf.predict(X_test)
   err_nn=np.linalg.norm(y_nn-y_test)/np.sqrt(tsize)
   lerr.append(err_nn) ##root mean square error on test data
   ytt_nn=clf.predict(tcts)
   lden.append(ytt_nn)  ##prediction for all training data
   covtest=np.mean(y_nn*y_test)-np.mean(y_nn)*np.mean(y_test)
   expynn=np.mean(y_nn*y_nn)-np.mean(y_nn)**2
   expytt=np.mean(y_test*y_test)-np.mean(y_test)**2
   if expynn < 1e-8: 
     pcctest=0
   else:
     pcctest=covtest/np.sqrt(expynn*expytt)
   lpcc.append(pcctest)
   covtrain=np.mean(yt_nn*y_train)-np.mean(yt_nn)*np.mean(y_train)
   expytnn=np.mean(yt_nn**2)-np.mean(yt_nn)**2
   expyttr=np.mean(y_train**2)-np.mean(y_train)**2
   if expytnn < 1e-8: 
     pcctrain=0
   else:
     pcctrain=covtrain/np.sqrt(expytnn*expyttr)
   lpcct.append(pcctrain)

  # Calculate mean and stdev
  den=np.array(lden)
  err=np.array(lerr)
  errt=np.array(lerrt)
  pcc=np.array(lpcc)
  pcct=np.array(lpcct)
  lind1.append(m)
  lind2.append(n)
  mden=np.mean(den,axis=0)
  merr=np.mean(err,axis=0)
  sden=np.std(den,axis=0)
  serr=np.std(err,axis=0)
  merrt=np.mean(errt,axis=0) #train
  serrt=np.std(errt,axis=0)  #train
  mpcc=np.mean(pcc,axis=0) #pcc-test
  spcc=np.std(pcc,axis=0)  #pcc-test
  mpcct=np.mean(pcct,axis=0) #pcc-train
  spcct=np.std(pcct,axis=0)  #pcc-train
  ltden.append(mden)
  lsden.append(sden)
  lterr.append(merr)
  lserr.append(serr)
  lterrt.append(merrt)
  lserrt.append(serrt)
  ltpcc.append(mpcc)
  lspcc.append(spcc)
  ltpcct.append(mpcct)
  lspcct.append(spcct)

ind1=np.array(lind1)
ind2=np.array(lind2)
tmden=np.array(ltden)
tsden=np.array(lsden)
tmerr=np.array(lterr)
tserr=np.array(lserr)
tmerrt=np.array(lterrt) #train
tserrt=np.array(lserrt) #train
tmpcc=np.array(ltpcc)
tspcc=np.array(lspcc)
tmpcct=np.array(ltpcct)
tspcct=np.array(lspcct)

print "shape(ind1)=",ind1.shape
print "shape(ind2)=",ind2.shape
print "shape(tmden)=",tmden.shape
print "shape(tmerr)=",tmerr.shape

###Output RMSE(test) (mean,stdev) versus Nh,Nc ###
f1=open("raw.test.results.dat","w")
tst=np.column_stack((ind1,ind2,tmerr,tserr,tmpcc,tspcc))
np.savetxt(f1, tst, fmt="%.3f")
f1.close()

###Output RMSE(train) (mean,stdev) versus Nh,Nc ###
f2=open("raw.train.results.dat","w")
tra=np.column_stack((ind1,ind2,tmerrt,tserrt,tmpcct,tspcct))
np.savetxt(f2, tra, fmt="%.3f")
f2.close()

fin=np.concatenate((dens.reshape(-1,1),tmden.reshape(-1,1),tsden.reshape(-1,1)),axis=1)
np.savetxt('mass.change.percent.NN.vs.pred.txt',fin,delimiter='\t')
