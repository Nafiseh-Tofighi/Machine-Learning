# Importing required libraries:
import numpy as np
import pandas as pd
import os

# Train and test .csv files have to be under source directory.
sourceDir='./'


# Read data into memory
# Features:
dataPoins = np.loadtxt(os.path.join(sourceDir,'hw01_data_points.csv'), 'str')
dataSet=[]
for line in dataPoins:
  dataSet.append(line.split(','))

# labels:
labels = np.loadtxt(os.path.join(sourceDir,'hw01_class_labels.csv'),'int')


# Spliting data
x_train=np.array(dataSet[:300])
x_test=np.array(dataSet[300:])

y_train=np.array(labels[:300])
y_test=np.array(labels[300:])

# Definning variables c and d as classes and featureLength, respectfully
featureLength=len(x_train[0])
classes=[1,2]

# Estimating the model parameters:

#==============pAcd===============
pAcd= np.empty((2,featureLength))

for d in range(featureLength):
  for c in classes:
    classC=x_train[y_train == c]
    pAcd[c-1, d]=len(classC[classC[:,d]=='A'])/len(classC)
print('\npAcd:')
print(pAcd)


#==============pCcd===============
pCcd= np.empty((2,featureLength))

for d in range(featureLength):
  for c in classes:
    classC=x_train[y_train == c]
    pCcd[c-1, d]=len(classC[classC[:,d]=='C'])/len(classC)
print('\npCcd:')
print(pCcd)


#==============pGcd===============
pGcd= np.empty((2,featureLength))

for d in range(featureLength):
  for c in classes:
    classC=x_train[y_train == c]
    pGcd[c-1, d]=len(classC[classC[:,d]=='G'])/len(classC)
print('\npGcd:')
print(pGcd)


#==============pTcd===============
pTcd= np.empty((2,featureLength))

for d in range(featureLength):
  for c in classes:
    classC=x_train[y_train == c]
    pTcd[c-1, d]=len(classC[classC[:,d]=='T'])/len(classC)
print('\npTcd:')
print(pTcd)


#==============class_priors===============
class_priors=[np.mean(y_train == (c + 1)) for c in range(len(classes))]
print('\nclass_priors:')
print(class_priors)


# Calculating the train set scores that are stored in variable 'g'
g=np.empty((len(classes),len(x_train)))

for i,n in enumerate(x_train):
  for c in classes:
    temp=1
    for d in range(featureLength):
      if n[d]=='A':
        temp*=pAcd[c-1,d]
      elif n[d]=='C':
        temp*=pCcd[c-1,d]
      elif n[d]=='G':
        temp*=pGcd[c-1,d]
      elif n[d]=='T':
        temp*=pTcd[c-1,d]
    g[c-1,i]=np.log(temp)+class_priors[c-1]

#Calculating the confusion matrix for the data points in training set
pred= np.array([int(np.argmax(g[:, r])+1)for r in range(g.shape[1])])
confusion_train = pd.crosstab(pred.T, y_train.T, rownames = ["y_pred"], colnames = ["y_truth"])


#==============confusion_train===============
print('\nconfusion_train:')
print(confusion_train)

# Calculating the test set scores that are stored in variable 'g_test'
g_test=np.empty((len(classes),len(x_test)))

for i,n in enumerate(x_test):
  for c in classes:
    temp=1
    for d in range(featureLength):
      if n[d]=='A':
        temp*=pAcd[c-1,d]
      elif n[d]=='C':
        temp*=pCcd[c-1,d]
      elif n[d]=='G':
        temp*=pGcd[c-1,d]
      elif n[d]=='T':
        temp*=pTcd[c-1,d]
    g_test[c-1,i]=np.log(temp)+class_priors[c-1]


#Calculating the confusion matrix for the data points in test set
pred_test= np.array([int(np.argmax(g_test[:, r])+1)for r in range(g_test.shape[1])])
confusion_test = pd.crosstab(pred_test.T, y_test.T, rownames = ["y_pred"],colnames = ["y_truth"])


#==============confusion_test===============
print('\nconfusion_test:')
print(confusion_test)