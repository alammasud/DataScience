import numpy as np
import pandas as pd
import random as rnd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


##############################################################
### This part reads the input file and scales the features
#############################################################
#############################################################

fp=open("red2.txt")
line=fp.readlines()
count=[]


bal=np.loadtxt(line,dtype='str').shape[1]

temlist=[]
cal=[]
y1=[]
for i in range(1,len(line)):
	temlist=[]
	for j in range(bal):
		if(j==bal-1):
			y1.append(np.float(line[i].split()[j]))
		else:
			temlist.append(np.float(line[i].split()[j]))
	cal.append(temlist)

yma=max(y1)
ymi=min(y1)
sumy=0.0
valu=[]
valu=[0 for i in range(len(line)-1)]


for i in range(len(line)-1):
	valu[i]=y1[i]/abs(yma)
	y1[i]=valu[i]


a=np.array(y1)[np.newaxis]
yy1=a.T
vv=np.insert(cal,0,1.0,axis=1)

xmax=vv.max(axis=0)
xmin=vv.min(axis=0)
diffn=xmax-xmin

for i in range(len(line)-1):
	for j in range(1,bal,1):
		vv[i][j]=vv[i][j]/abs(xmax[j])


##################################################
######### This part is about normal method of linear regression
###############################################
xtr=np.transpose(vv)
BB=np.matmul(xtr,vv)
A=np.linalg.inv(BB)
B=np.matmul(A,xtr)
theta=np.matmul(B,yy1)

print("\n")
print("Normal method\n")
print("\n")
for i in range(0,bal-1):
	print(line[0].split()[i],'\t' ,*theta[i+1])


###########################################
## Here is the code with scikit learn library
############################################

X_train, X_test, y_train, y_test = train_test_split(vv, yy1, test_size=0.1, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("\n")
print("Scikit built in library\n")
print("\n")
print(regressor.coef_)

###################################
## Here is the gradient descent method
####################################

theta=np.random.rand(bal,1)

alpha=0.01
itrnum=1000000
for i in range(itrnum):
	hx=np.matmul(vv,theta)
	diff=hx-yy1
	new=np.array(np.sum(np.transpose(diff)@vv,axis=0))
	tt=np.reshape(new,(bal,1))
	theta=theta-alpha*tt/len(yy1)

print("\n")
print("Here is with Gradident descent\n")

for i in range(0,bal-1):
	print(line[0].split()[i],'\t' ,*theta[i+1])
