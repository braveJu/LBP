import numpy as np
import cv2
from copy import deepcopy
def mat_access1(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            k = mat[i, j]
            mat[i, j] = k * 10
def mat_access2(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            k = mat.item(i, j)
            mat.itemset((i, j), k * 2)


def mat_access3(mat, c = 1, r = 0.3):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            k = mat.item(i, j)
            mat[i, j] = c * np.power(k, r)


def mat_access4(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            k = mat.item(i, j)
            mat[i,j] = 255- k

mat1 = np.arange(10).reshape(2, 5)
mat2 = np.arange(10).reshape(2, 5)
print
print("원소 처리 전 : ")
print(mat1)
print("원소 처리 후 ")
mat_access1(mat1)
print(mat1)
print("원소 처리 전 : ")
print(mat2)
print("원소 처리 후 ")
mat_access2(mat2)
print(mat2)


def flip(image, flip_code = 'x'):
    cp_image = deepcopy(image)
    h, w = image.shape
    for i in range(ima)
