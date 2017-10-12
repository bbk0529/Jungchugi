# Produced on 25th Sep 17'
# Python
# ML - Deep Learning 




#Perceptron only recieved zero(0) or one (1) for inputs 

def AND(x1,x2):
    w1, w2,theta = 0.5, 0.5, 0.7  
    if x1*w1+x2*w2<=theta :   #classifier
        return 0
    else :
        return 1


# AND function revised to implement bias theroy

import numpy as np

def AND_bias (x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    if np.sum(w*x) + b <= 0 :
        return 0
    else : 
        return 1


# AND FUNCTION TEST
for xs in [(0,0), (0,1), (1,0), (1,1)]:
    y = AND(xs[0], xs[1])
    print(str(xs)+ " -> "+ str(y))
	
	
	

def NAND (x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    if np.sum(w*x) + b <= 0 :
        return 0
    else : 
        return 1
		

# NAND FUNCTION TEST
for xs in [(0,0), (0,1), (1,0), (1,1)]:
    y = NAND(xs[0], xs[1])
    print(str(xs)+ " -> "+ str(y))
	
	
	

def OR (x1,x2):
    x = np.array([x1,x2])
    w = np.array([1.0, 1.0])
    b = -0.5
    if np.sum(w*x) + b <= 0 :
        return 0
    else : 
        return 1
		


# NAND FUNCTION TEST
for xs in [(0,0), (0,1), (1,0), (1,1)]:
    y = OR(xs[0], xs[1])
    print(str(xs)+ " -> "+ str(y))
	

#XOR is not made of a sigle layer
def XOR (x1,x2):
	return AND (NAND(x1,x2), OR(x1,x2))    

for xs in [(0,0), (0,1), (1,0), (1,1)]:
    y = XOR(xs[0], xs[1])
    print(str(xs)+ " -> "+ str(y))
	
	

	
#STEP FUNCTION
# h(x) = 0 (x<=0)	
	
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	#return np.array(x>0, dtype=np.int)
	return np.array(x>0)

x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y, 'g--')
#plt.ylim(-0.1,1.1)
plt.show()

#Sigmoid Function

def sigmoid(x):
	return 1/(1+exp(-x))
	
x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()	

#reLu Function

def relu(x):
	return np.maximum(0,x)
	
x=np.arange(-5.0,5.0,0.1)
y=relu(x)
plt.plot(x,y, 'r.')
plt.ylim(-0.1,5)
plt.show()	


def identity_function(x):
	return x


X=np.array([1.0,0.5])
			   
W1 = np.array([[0.1,0.2],
			   [0.3,0.4],
			   [0.5,0.6]])
   
B1 = np.array([0.1,0.2,0.3])
A1=np.dot(W1,X) + B1; print(A1)
Z1=sigmoid(A1); Z1

W2 = np.array([[0.1,0.2,0.3],
               [0.4,0.5,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(W2,Z1) + B2; print(A2)
Z2 = sigmoid(A2);Z2


W3 = np.array([[0.1,0.2],
			   [0.3,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(W3,Z2) + B3; print(A3)
Y = identity_function(A3); print(Y) #Y = Z3



def init_network():
	network= {}
	network['W1'] =  np.array([[0.1,0.2],
			                   [0.3,0.4],
			                   [0.5,0.6]])	
	network['W2'] = np.array([[0.1,0.2,0.3],
                             [0.4,0.5,0.6]])						 	
	network['W3'] = np.array([[0.1,0.2],
			                 [0.3,0.4]])
	
	network['B1'] = np.array([0.1,0.2,0.3])	
	network['B2'] = np.array([0.1,0.2])
	network['B3'] = np.array([0.1,0.2])
	return network
							 
def forward(network, X): 
	W1,W2,W3 = network['W1'], network['W2'], network['W3']
	B1,B2,B3 = network['B1'], network['B2'], network['B3']
	
	A1=np.dot(W1, X) + B1
	Z1=sigmoid(A1)
	A2=np.dot(W2,Z1) + B2
	Z2=sigmoid(A2)	
	A3=np.dot(W3,Z2) + B3
	Y=identity_function(A3)
	print(Y)
	
X=np.array([1.0,0.5])

init_network()
forward(network,X)



#SOFTMAX FUNC. normalized effect 

a=np.array([1000,990, 1010])
def softmax(x):
	return exp(x-max(a)) / (sum(exp(x)-max(a)))
