import tensorflow as tf
import numpy as np

"""
tensorflow 定义了整套数据结构代替 python 原生数据类型进行计算 
"""

print("\r\n************************* constant **************************")
# 常量;dtype = int32
print("constant: ",tf.constant(1))

# 常量;dtype = float32
print("constant: ",tf.constant(1.))

# 常量字符串;dtype = string
print("constant string: ",tf.constant("constant string"))

# 常量数组;dtype = int32
print("constant vector: ",tf.constant([1,2,3]))

print("\r\n************************* Variable **************************")
# 变量 dtype=int32
print("variable: ",tf.Variable(1))

# 名字为 v1 的变量 dtype=float32
print("variable: ",tf.Variable(1.0,name="v1"))

# 变量 dtype=string
print("variable string: ",tf.Variable("constant string"))

# 数组变量 dtype=int32
print("variable vector: ",tf.Variable([1,2,3]))

print("\r\n************************* cast **************************")
a = tf.range(5)
print("int to float: ",tf.cast(a,dtype = tf.float32))

a = [0,1]
print("int to boolean: ",tf.cast(a,dtype = tf.bool))

print("\r\n************************* to tensor **************************")
# python 类型转换成 tensor
a = 1
print("int to tensor: ",tf.convert_to_tensor(a))

a = np.arange(5)
print("vector to tensor: ",tf.convert_to_tensor(a,dtype = tf.float32))


print("\r\n************************* to numpy **************************")
# tensorflow 里数据类型转化为 python 类型
a = tf.Variable(1)
print("int tensor to numpy: ",a.numpy())

a = tf.range(5)
print("vector tensor to numpy: ",a.numpy())