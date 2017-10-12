## Argument amendments. 
## 28th Sep 17'
#common.layers.py
#ch7.simple_convnet
pwd
cd /pydev/NN_Shared/ch06

import optimizer_compare_naive
import optimizer_compare_mnist


#im2col 함수 사용예
import sys, os
sys.path.append(os.pardir)
from common.util import im2col

x1=np.random.rand(1,3,7,7)  #(데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1,5,5, stride=1, pad=0)
print(x1.shape)
print(col1.shape)



#numpy의 reshape
 x=np.array(np.arange(32))
 
 x.reshape((3,4))
 # ValueError: cannot reshape array of size 32 into shape (3,4)
 
 In [37]: x.reshape((4,-1))
Out[37]:
'''
array([[ 0,  1,  2,  3,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29, 30, 31]])
'''
# -1 은 나머지 입력에 맞추어 알아서 맞출것.    -1은 하나만쓸 수 있다. 
x.reshape(2, -1, 8 )

f=np.array(np.arange(27)).reshape(1,3,3,3)
f.reshape(1,-1).T 

f=np.array(np.arange(54)).reshape(2,3,3,3)
f.reshape(2,-1).T  #filters 2 ea simulated 

x=np.array(np.arange(108)).reshape(-1,27)
f2=f.reshape(2,-1)
print(x.shape)
print(f2.T.shape)

out=np.dot(x,f2.T)
out.reshape(1,2,2,-1).transpose(0,3,1,2)



# deep learning from scratch, 
################################################################
'''
ch07.visualize_filter.py
ch07.train_convnet
'''


