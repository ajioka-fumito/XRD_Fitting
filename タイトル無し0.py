# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:52:23 2020

@author: Adachi-ab
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,16,1.5)
y = 3*x + 1

plt.scatter(x,y)

import scipy.interpolate as intersp

yy = intersp.spline(x,y,np.arange(0,15,1))

plt.scatter(np.arange(0,15,1),yy)
plt.show()
print(yy)