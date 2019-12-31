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

fp=open("datalog.txt")
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

#yma=max(y1)
#ymi=min(y1)
#sumy=0.0
#valu=[]
#valu=[0 for i in range(len(line)-1)]


#for i in range(len(line)-1):
#	valu[i]=y1[i]/abs(yma)
#	y1[i]=valu[i]


a=np.array(y1)[np.newaxis]
yy1=a.T
vv=np.insert(cal,0,1.0,axis=1)
#vv=np.array(cal)
#bal=bal-1

xmax=vv.max(axis=0)
xmin=vv.min(axis=0)
diffn=xmax-xmin

for i in range(len(line)-1):
	for j in range(1,bal,1):
		vv[i][j]=vv[i][j]/abs(xmax[j])



def sigmoidfun(X,theta):
	z=np.matmul(X,theta)
	bp=1.0/(1+np.exp(-z))
	return bp



###################################
## Here is the gradient descent method
####################################

theta=np.random.rand(bal,1)

alpha=0.01
itrnum=1000000
for i in range(itrnum):
	hx=sigmoidfun(vv,theta)
	diff=hx-yy1
	new=np.array(np.sum(np.transpose(diff)@vv,axis=0))
	tt=np.reshape(new,(bal,1))
	theta=theta-alpha*tt/len(yy1)

print("\n")
print("Here is with Gradient descent\n")

for i in range(0,bal-1):
	print(line[0].split()[i],'\t' ,*theta[i+1])

#print(yy1)
newvv=np.reshape(cal,(len(line)-1,bal-1))

xnn=np.c_[newvv,yy1]



xpn=xnn.T
#print(xpn[0])
#print(xpn[1])
xxx1=np.arange(0,10,0.1)
xxx2=-(theta[0] + theta[1]*xxx1)/theta[2]

plt.scatter(xpn[0][0:50], xpn[1][0:50], color='r', label='1')
plt.scatter(xpn[0][50:100], xpn[1][50:100], color='b', label='0')
plt.plot(xxx1, xxx2, c='k', label='Decision')
plt.show()
#plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 
#plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 
