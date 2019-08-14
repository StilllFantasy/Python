import cv2
import matplotlib.pyplot as plt 
import PIL
import numpy as np
import random
path = 'D:\\Python\\打码与解码\\sunjia.jpg'

image = PIL.Image.open(path)
data = np.array(image)
x = data.shape[0]
y = data.shape[1]
k = int((x+y) / 100)
for i in range(0, x, k):
    for j in range(0, y, k):
        if i + k >=x or j + k >= y:
            break
        num = data[i][j]
        for xx in range(i, i + k):
            for yy in range(j, j + k):
                data[xx][yy] = num 
plt.imshow(data)
plt.show()