##### 
#####
####

import numpy as np
#평균제곱오차
t= [0,0,1,0,0,0,0,0,0,0]
# 예1 : '2'일 확률이 가장 높다고 추정
y = [0.1,0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]


def mean_squared_error(y,t) : 
	return 0.5 * np.sum((y-t)**2)

mean_squared_error(np.array(y), np.array(t))

# 예1-1 : '2'일 확률이 가장 높다고 추정된 결과, 예2보다 값이 낮게 나온다. 
y = [0.1,0.05,0.7,0.0,0.05,0.05,0.0,0.05,0.0,0.0]
mean_squared_error(np.array(y), np.array(t))

# 예1-2 : '7'일 확률이 가장 높다고 추정된 결과
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_squared_error(np.array(y), np.array(t))


#교차 엔트로피 오차
def cross_entroypy_error(y,t):
	delta = 1e-7
	return -np.sum(t * np.log(y+delta))

t= [0,0,1,0,0,0,0,0,0,0]
# 예2-1 : '2'일 확률이 가장 높다고 추정된 결과, 예2보다 값이 낮게 나온다. 
y = [0.1,0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entroypy_error(np.array(y), np.array(t))

# 예2-2 : '2'일 확률이 가장 높다고 추정된 결과
y = [0.1,0.05,0.7,0.0,0.05,0.1,0.0,0.0,0.0,0.0]
cross_entroypy_error(np.array(y), np.array(t))



# 예1-2 : '7'일 확률이 가장 높다고 추정된 결과
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
cross_entroypy_error(np.array(y), np.array(t))



# (배치용) 교차 엔트로피 오차
#정답 레이블이 원-핫 인코딩인 경우

def cross_entroy_error(y,t) :
	if y.ndim ==1:
		t=t.reshape(1,t.size)
		y=y.reshape(1,y.size)
	
	batch_size=y.shape[0]
	return -np.sum(t*np.log(y)) / batch_size
	
############################	
np.random.choice(1000,10)
np.random.choice(100,5)

import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) =load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)

train_size = x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size) #Using np.random.choice, to pick 10 samples randomly from train_size 

print (batch_mask)


x_batch = x_train[batch_mask]
t_batch=t_train[batch_mask]

print(x_batch[0])
print(t_batch[0])

### Cost Function / Loss Function 
### (배치용) 교차 엔트로피 오차 구현하기
def cross_entroy_error (y,t):
	if y.ndim ==1 :
		t = t.reshape(1,t.size)
		y = y.reshape(1,y.size)
	
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y)) / batch_size



import neuralnet_mnist	
network = neuralnet_mnist.init_network()

y=neuralnet_mnist.predict(network, x_batch)
print(y)
print(t_batch)

cross_entroy_error(y,t_batch)
y=np.arange(30).reshape(3,10)
print(y)

#수치해석 - Numerical Analytic 
#수치미분 - y=ax^2 → y`=2ax


np.float32(1e-50) 
# 결과가 0이 나온다. - 반올림 문제가 있다. 1e-4 사용한다.
# 차분과 기울기의 간극, 원인 전방차분, 해결책 중앙 차분 

def numerical_diff(f,x,h=1e-4):	
	return(f(x + 2*h) - f(x)) / (2*h)


	
def function_1(x):
	return 0.01*x**2 + 0.1*x
	
x=np.arange(0.0,20.0,0.1)	
y=function_1(x)
plt.plot(x,y)

numerical_diff(function_1,5)

def tangent_line(f, x, h=1e-4):
    a = numerical_diff(f, x, h)
    print(a)
    b = f(x) - a*x
    return lambda t: a*t + b ## lambda 함수를 return함 
	
	
tangent=tangent_line(function_1,5) #tangent는 5지점에서의 기울기를 가리고 있는 1차함수 
plt.xlabel('x');plt.ylabel('f(x)')

plt.plot(x,function_1(x))
plt.plot(x,tangent_line(function_1,5)(x))

for i in range(1,10):
	plt.plot(x,tangent_line(function_1,5,h=i)(x))

	
#### CH04	
# enumerate
for idx,x in enumerate(['a','b','c']):
	print(idx,x)
	
	
import(gradient_2d)
%run gradient_2d

def function_2(X):
	return X[0]**2 + X[1]**2 ## f(x)= x0^2 + x1^2

init_x=np.array([-3.0,4.0]) ##시작위치 

gradient_method.gradient_descent(function_2, init_x, lr=0.1, step_num=100)

init_x=np.array([-3.0,-4.0]) ##시작위치

a,b=gradient_method.gradient_descent(function_2, init_x, lr=0.1, step_num=100)

plt.scatter(x=b.transpose()[0],y=b.transpose()[1])



########################################################
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x  #shallow copy로 x의 변화가 init_x의 영향을 미친다. 
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

	
#파이썬 리스트 복사 / shallow copy, deep copy
a=[9,2,1,4]
b=a
print(a,b)

b[0]=999
print(a,b)

c=a[:]
d=list(a)
print (c is a)
print (d is a)
print (b is a)

a=[1,2,['a','b'],3]
c=a[:]
print(c is a)

a[2][1]='xyz'
print(a)
print(c)

import copy
d=copy.deepcopy(a)
a[2][0] = 'abc'
print(a)
print(c)
print(d)
a[0]=111
print(a)
print(c)
print(d)

def func_a(x):
	my_x = x
	print(my_x)
	my_x = 200
	return my_x

a=[1,2,3] # 참조된다. list는 참조 값이 넘어가면서, 내부에서 수정한 값이 외부에 영향을 준다. 
def func_a(x):
	x[0]=0
	return x[0] 
	
b=(1,2,3)


c=a[:]  # 참조된다. 
#c와 a가 다른 것처럼 응답하지만, false , 실제로는 참조한다. 
print (c is a)
c[0]=111
print(c)
print(a)

d=np.array(a) ## 복제된다. 
print (d is a)
d[1] = 222
print(d)
print(a)

e=a.copy() ### 복제된다. 
e[2] = 333
print(e)
print(a)

################### numpy array 객체복사 ####################
a=np.array([1,2,3,4])



##############################################################



import gradient_method

for i in range (-10,10):
	for j in range(-10,10) :
		init_x=np.array([float(i),float(j)]) ##시작위치
		a,b=gradient_method.gradient_descent(function_2, init_x, lr=0.1, step_num=100)
		plt.scatter(x=b.transpose()[0],y=b.transpose()[1])
		
		
		
################################################################

# 신경망에서의 기울기를

import gradient_simplenet
import train_neuralnet

#train acc, test acc | 0.112383333333, 0.1135
#train acc, test acc | 0.784283333333, 0.7889
#train acc, test acc | 0.875933333333, 0.8792
#train acc, test acc | 0.89885, 0.9008
#train acc, test acc | 0.9085, 0.9128
#train acc, test acc | 0.914966666667, 0.9178
#train acc, test acc | 0.919183333333, 0.9215
#train acc, test acc | 0.9231, 0.9248
#train acc, test acc | 0.9267, 0.9287
#train acc, test acc | 0.930466666667, 0.9318
#train acc, test acc | 0.933666666667, 0.9338
#train acc, test acc | 0.936966666667, 0.9363
#train acc, test acc | 0.939516666667, 0.938
#train acc, test acc | 0.9419, 0.941
#train acc, test acc | 0.9439, 0.9433
#train acc, test acc | 0.945733333333, 0.944
#train acc, test acc | 0.947316666667, 0.9458
