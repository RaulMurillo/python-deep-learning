import numpy as np
from demo_code.util import im2col, im2col2
#from demo_code.layers import Convolution
import timeit

input_matrix = np.array([[[[3,9,0], [2, 8, 1], [1,4,8]]]])
# print(input_matrix)

B = np.array([[[[ 0,  1,  2], [ 4,  5,  6], [ 8,  9, 10], [12, 13, 14]], [[ 1,  2,  3], [ 5,  6,  7], [ 9, 10, 11], [13, 14, 15]]], [[[ 4,  5,  6], [ 8,  9, 10], [12, 13, 14], [16, 17, 18]], [[ 5,  6,  7], [ 9, 10, 11], [13, 14, 15], [17, 18, 19]]]])
# print(B)

kernel = np.array([[8,9], [4,4]])
# print(kernel)

im2col_matrix = im2col(input_matrix, kernel.shape[0], kernel.shape[1])
# print(im2col_matrix)

windows = im2col2(input_matrix, kernel.shape[0], kernel.shape[1])
# print(windows)

assert np.equal(im2col_matrix, windows).all()


im2col_matrix = im2col(B, kernel.shape[0], kernel.shape[1])
# print(im2col_matrix)

windows = im2col2(B, kernel.shape[0], kernel.shape[1])
# print(windows)

assert np.equal(im2col_matrix, windows).all()

print("OK!")

x = np.random.rand(4, 6, 100, 100)
K = np.random.rand(8, 6, 3, 3)

pad = 2
stride = 3

FN, C, FH, FW = K.shape
N, C, H, W = x.shape
out_h = (H + 2*pad - FH) // stride + 1
out_w = (W + 2*pad - FW) // stride + 1

#1: Naive approach
col1 = im2col(x, FH, FW, stride, pad)
print(col1.shape)
col_W = K.reshape(FN, -1).T

out1 = np.dot(col1, col_W) # + self.b

#2: Memory Strides
col2 = im2col2(x, FH, FW, stride, pad)
print(col2.shape)
# col_W = W.reshape(FN, -1).T

out2 = np.dot(col2, col_W)

assert np.equal(out1, out2).all()
print("OK!")

def foo():
    col1 = im2col(x, FH, FW, stride, pad)
    out1 = np.dot(col1, col_W)
    return out1

def bar():
    col2 = im2col2(x, FH, FW, stride, pad)
    out2 = np.dot(col2, col_W)
    return out2

# for n in range(1, 11):
#     print("n =", n)
#     x = np.random.rand(4, 6, 100*n, 100*n)
#     K = np.random.rand(8, 6, 3, 3)
#     FN, C, FH, FW = K.shape
#     col_W = K.reshape(FN, -1).T

#     t1 = timeit.timeit(foo, number=100)
#     print(" Time for im2col:", t1)
#     t2 = timeit.timeit(bar, number=100)
#     print(" Time for im2col2:", t2)
#     print("Speed-up: ", t1/t2)
#     print("*********")

x = np.random.rand(1, 1, 100, 100)
K = np.random.rand(1, 1, 3, 3)
FN, C, FH, FW = K.shape
col_W = K.reshape(FN, -1).T

t1 = timeit.timeit(foo, number=100)
print(" Time for im2col:", t1)
t2 = timeit.timeit(bar, number=100)
print(" Time for im2col2:", t2)
print("Speed-up: ", t1/t2)
print("*********")

x = np.random.rand(1, 1, 5000, 5000)
K = np.random.rand(1, 1, 3, 3)
FN, C, FH, FW = K.shape
col_W = K.reshape(FN, -1).T

t1 = timeit.timeit(foo, number=100)
print(" Time for im2col:", t1)
t2 = timeit.timeit(bar, number=100)
print(" Time for im2col2:", t2)
print("Speed-up: ", t1/t2)
print("*********")