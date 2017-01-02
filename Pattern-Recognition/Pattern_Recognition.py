import matplotlib as plt
import numpy as np

from sklearn import svm
from skimage import color
from skimage import io

svc = svm.SVC(gamma=0.05, cache_size=100)

pen = color.rgb2gray(io.imread('images/pen.jpg'))[...,2];
pen_2 = color.rgb2gray(io.imread('images/pen2.jpg'))[...,2];
pen_3 = color.rgb2gray(io.imread('images/pen3.jpg'))[...,2];
car = color.rgb2gray(io.imread('images/car.jpg'))[...,2];
car_2 = color.rgb2gray(io.imread('images/car2.png'))[...,2];
notebook = color.rgb2gray(io.imread('images/notebook.png'))[...,2];

input, target = [pen, car, notebook], ['pen', 'car', 'notebook']
svc.fit(input, target)

print 'Prediction: %s' % svc.predict(pen)[0]
print 'Prediction: %s' % svc.predict(pen_2)[0]
print 'Prediction: %s' % svc.predict(pen_3)[0]
print 'Prediction: %s' % svc.predict(car)[0]
print 'Prediction: %s' % svc.predict(car_2)[0]
print 'Prediction: %s' % svc.predict(notebook)[0]