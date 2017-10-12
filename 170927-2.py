####
###
# REFERENCE
# train_neuralnet.py
# ch05/two_layer_net.py

# 순전파 순서대로 집어넣겠다. ordered Dict 에 보관할 것임 중요. 



import numpy as np


X=np.random.rand(2) 	# 입력 (훈련데이터)은 한건인데, 2차원 데이터라 원소가 2건임을 유의 ex) 종양크기, 나이
W=np.random.rand(2,3) 	# 가중치
B=np.random.rand(3)

X.shape
(2,)

W.shape
(2, 3)

B.shape
(3,)




########################
# 숫자 해독 실습 #######
########################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


(x_train, t_train) , (x_test,t_test) = load_mnist(normalize = True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


x_batch = x_train[:3]
t_batch = t_train[:3]

print(t_batch)
print(x_batch.shape)

#gradient = backpropagation
#numerical_gradient = 수치미분
grad_numerical = network.numerical_gradient(x_batch,t_batch)
grad_backprop = network.gradient(x_batch,t_batch)

for key in grad_numerical.keys():
	diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
	print(key + ":" + str(diff))

	
	
### 작성한 신경망
import train_neuralnet


#계산그래프
import numpy as np
import matplotlib.pylot as plt

x=np.arange(len(train_neuralnet.train_acc_list))
plt.plot(x, train_neuralnet.train_acc_list)
plt.plot(x, train_neuralnet.test_acc_list)

x=np.arange(len(train_neuralnet.train_loss_list))
print(len(x))
plt.plot(x, train_neuralnet.train_loss_list)