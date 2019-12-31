import numpy as np
import pandas as pd
import random as rnd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



fp=open("new.txt")
line=fp.readlines()
count=[]


bal=np.loadtxt(line,dtype='str').shape[1]

print(bal) 

temlist=[]
cal=[]
y1=[]
x1=[]
x2=[]
for i in range(1,len(line)):
	for j in range(bal):
		if(j==bal-1):
			x2.append(np.float(line[i].split()[j]))
		else:
			x1.append(np.float(line[i].split()[j]))


nc=2
x1max=max(x1)
x1min=min(x1)

x2max=max(x2)
x2min=min(x2)

xc1=x1[3]
yc1=x2[3]
#xc1=0.0
#yc1=0.0

xc2=x1[len(line)-10]
yc2=x2[len(line)-10]
#xc2=91
#yc2=86

#xclus1=[]
#xclus2=[]

#xcor1=[]
#ycor1=[]

#xcor2=[]
#ycor2=[]

for j in range(500):
	xclus1=[]
	xclus2=[]

	xcor1=[]
	ycor1=[]

	xcor2=[]
	ycor2=[]
	for i in range(len(line)-1):
		dx1=x1[i]-xc1
		dy1=x2[i]-yc1
		
		dx2=x1[i]-xc2
		dy2=x2[i]-yc2
		
		d1=math.sqrt(dx1*dx1+dy1*dy1)
		d2=math.sqrt(dx2*dx2+dy2*dy2)
		
		if(d1<d2):
			xclus1.append(i)
		elif(d2<d1):
			xclus2.append(i) 
	ttx=0.0
	tty=0.0
	
	for i in range(len(xclus1)):
		pp=xclus1[i]
		xcor1.append(x1[pp])
		ycor1.append(x2[pp])

		ttx=ttx+x1[pp]
		tty=tty+x2[pp]
	
	tlnx=0.0
	tlny=0.0
	
	for i in range(len(xclus2)):
		ll=xclus2[i]
		xcor2.append(x1[ll])
		ycor2.append(x2[ll])

		tlnx=tlnx+x1[ll]
		tlny=tlny+x2[ll]
		
	xc1=ttx/len(xclus1)
	yc1=tty/len(xclus1)
		
	xc2=tlnx/len(xclus2)
	yc2=tlny/len(xclus2)
	
	print(xc1,yc1)
	print(xc2,yc2)
	print(len(xclus1),len(xclus2))



plt.scatter(xcor1, ycor1, color='r', label='1')
plt.scatter(xcor2, ycor2, color='b', label='0')
plt.scatter(xc1,yc1, color='k', label='cir1')
plt.scatter(xc2,yc2, color='k', label='cir2')
plt.show()
