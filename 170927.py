## CH05 - Backward Propagation 
## 27th Sep 17'
# Apple shopping - 

import numpy as np
from layer_naive import *


###############################################
# Example, a shopping for apples  only        #
###############################################
apple_price=100
apple_num=2
tax = 1.1

#Nodes for mul prepared,
mul_apple_node = MulLayer()
mul_tax_node = MulLayer()


#Forward Propagation
apple_total_price = mul_apple_node.forward(apple_price,apple_num)
total_price = mul_tax_node.forward(apple_total_price, tax)


#Backward Propagation
d_total_price = 1 #d means 'derivative'
d_apple_total_price, d_tax = mul_tax_node.backward(d_total_price)
d_apple_price, d_apple_num = mul_apple_node.backward(d_apple_total_price)




###############################################
# Example, a shopping for oranges and apples  #
###############################################

#initiations of variables
apple_num=2
apple_price=100
orange_num=3
orange_price=150
tax=1.1

# preparation of nodes
mul_apple_node = MulLayer()
mul_orange_node = MulLayer()
mul_tax_node = MulLayer()
add_fruit_node = AddLayer()


#forward propagation
apple_total_price = mul_apple_node.forward(apple_price,apple_num)
orange_total_price = mul_orange_node.forward(orange_price,orange_num)
total_price = add_fruit_node.forward(apple_total_price, orange_total_price)
taxed_total_price = mul_tax_node.forward(total_price,tax)

print('Price without Tax	', total_price)
print('price with Tax		', taxed_total_price)


#backward propagation
d_taxed_total_price = 1
d_total_price,d_tax =mul_tax_node.backward(d_taxed_total_price)
d_apple_total_price, d_orange_total_price = add_fruit_node.backward(d_total_price)
d_apple_price,d_apple_num = mul_apple_node.backward(d_apple_total_price)
d_orange_price,d_orange_num = mul_orange_node.backward(d_orange_total_price)


print('Derivative of orange price	',d_orange_price)
print('Derivative of orange num		',d_orange_num)
print('Derivative of apple price	',d_apple_price)
print('Derivative of apple num		',d_apple_num)