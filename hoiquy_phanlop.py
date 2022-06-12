import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#from sklearn import linear_model
#import keras
x = np.array([[ 7.79667589],
[ 2.79825217],
[ 2.06174503],
[ 4.4713877 ],
[ 7.20443649],
[ 7.36014312],
[ 4.70688117],
[-0.40338389],
[ 4.72266607],
[ 1.20453709],
[ 6.07593449],
[ 7.69651292],
[ 3.89733971],
[ 4.7856351 ],
[-0.59932188],
[ 4.1507473 ],
[ 0.04186784],
[ 4.89562846],
[ 2.38650347],
[ 6.42758034]])

y =np.array([[ 318.28185696],
[ 20.48143891],
[ 11.97873995],
[ 7.56902114],
[ 224.15497306],
[ 235.04403786],
[ 17.75040067],
[-107.86335911],
[ 1.1140603 ],
[ -7.67492972],
[ 87.4263873 ],
[ 293.22569099],
[ -11.49557421],
[ 6.4415876 ],
[-152.88870565],
[ -4.95755333],
[ -79.53431819],
[ 34.97246059],
[ -4.50098315],
[ 95.09276699]])

#def predict(new_x,w1,w0):
plt.figure(figsize=(8, 6))
plt.scatter(x,y, marker = 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot của x và y')
plt.show()

x_poly = PolynomialFeatures(degree=3).fit_transform(x)
reg = LinearRegression().fit(x_poly, y)
x_interval = np.arange(np.min(x), np.max(x) + 0.1, 0.1).reshape(-1, 1)
x_interval_poly = PolynomialFeatures(degree=3).fit_transform(x_interval)
y_interval = reg.predict(x_interval_poly)
plt.figure(figsize=(8, 6))
plt.plot(x_interval, y_interval, 'r')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kết quả hồi quy đa thức bậc 3')
plt.show()