import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm

def data_to_image(sample):
    matriz = []
    array = []
    for x in xrange(0, 64):
        array.append(sample[x])
        if (x+1)%8 == 0:
            matriz.append(array)
            array = []
    return np.array(matriz)

# Take digit pictures
digits = datasets.load_digits()

# Classifer
svc = svm.SVC(gamma=0.05, C=100)

x_data = np.array([16,0,0,0,0,0,0,16,
                      0,16,0,0,0,0,16,0,
                      0,0,16,0,0,16,0,0,
                      0,0,0,16,16,0,0,0,
                      0,0,0,16,16,0,0,0,
                      0,0,16,0,0,16,0,0,
                      0,16,0,0,0,0,16,0,
                      16,0,0,0,0,0,0,16])

o_data = np.array([16,16,16,16,16,16,16,16,
                     16,0,0,0,0,0,0,16,
                     16,0,0,0,0,0,0,16,
                     16,0,0,0,0,0,0,16,
                     16,0,0,0,0,0,0,16,
                     16,0,0,0,0,0,0,16,
                     16,0,0,0,0,0,0,16,
                     16,16,16,16,16,16,16,16])

# Training
x, y = [x_data, o_data], ['x', 'o']
svc.fit(x,y)
        

# Prediction
print('Prediction: ', svc.predict(x_data)[0])
plt.imshow(data_to_image(x_data), cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()