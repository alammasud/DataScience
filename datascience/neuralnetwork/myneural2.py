import numpy as np


def sigmoid(x):
	bal=1.0/(1.0+np.exp(-x))
	return bal

def derivative(x):
	return x*(1-x)

nin=2
nhid=2
nou=1

#numitr=1
lrate=0.3

X=np.array([[0,0,1,1],[0,1,0,1]])
Y=np.array([[0,1,1,0]])
m=X.shape[1]

theta1=[]
theta2=[]

bias1=[]
bias2=[]

theta1=np.random.randn(nhid,nin)
theta2=np.random.randn(nou,nhid)
bias1=np.ones((nhid,1))
bias2=np.ones((nou,1))


numitr=10000
        
cost=1.0
while(cost>0.005):
	Z1=np.dot(theta1,X)+bias1
	A1=sigmoid(Z1)
	Z2=np.dot(theta2,A1)+bias2
	A2=sigmoid(Z2)
	
	dz2=A2-Y
	dtheta2=np.dot(dz2,A1.T)/m
	db2=np.sum(dz2, axis=1, keepdims=True)/m
	dz1=np.multiply(np.dot(theta2.T, dz2), 1-np.power(A1, 2))
	dtheta1=np.dot(dz1, X.T)/m
	db1 = np.sum(dz1, axis=1, keepdims=True)/m
	
	cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2)))/m
	
	theta1=theta1-lrate*dtheta1
	bias1=bias1-lrate*db1
	theta2=theta2-lrate*dtheta2
	bias2=bias2-lrate*db2
	print(cost)

X=np.array([[1],[1]])
Z1=np.dot(theta1,X)+bias1
A1=sigmoid(Z1)
Z2=np.dot(theta2,A1)+bias2
A2=sigmoid(Z2)
if(A2>=0.5):
	print(1)
else:
	print(0)
