# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:03:02 2021

@author: Enqey De-Ben Rockson
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 

rng = np.random.RandomState(1)

x = np.dot(rng.rand(2,2), rng.rand(2,200)).T
plt.scatter(x[:,0],x[:,1])
plt.axis('equal')
plt.show()

pca = PCA(n_components=2)
pca.fit(x)

def     draw_vector(v0, v1, ax =None):
        ax = ax or plt.gca()
        arrowprops = dict(arrowstyle = '->',
                              linewidth = 2,
                              shrinkA = 0, shrinkB = 0)
        ax.annotate('', v1, v0, arrowprops=arrowprops)

plt.scatter(x[:,0],x[:,1])
for lenght, vector in zip (pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(lenght)
    draw_vector(pca.mean_,pca.mean_ + v)
plt.axis('equal')
plt.show()


pca = PCA(n_components =1)
pca.fit(x)
x_pca = pca.transform(x)
x_new = pca.inverse_transform(x_pca)
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(x_new[:, 0], x_new[:, 1], alpha=(0.8))
plt.axis('equal')
plt.show()