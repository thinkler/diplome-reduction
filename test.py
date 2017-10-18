import numpy as np
import pylab as pl
import pdb

def andrews_curve(x,theta):
  curve = list()

  for th in theta:
    x1 = x[0] / np.sqrt(2)
    x2 = x[1] * np.sin(th)
    x3 = x[2] * np.cos(th)
    x4 = x[3] * np.sin(2.*th)
    # pdb.set_trace()
    curve.append(x1+x2+x3+x4)
  return curve

accuracy = 1000
samples = np.loadtxt('iris.csv', usecols=[0,1,2,3], delimiter=',')
theta = np.linspace(-np.pi, np.pi, accuracy)

for s in samples[:1]: # setosa
  pl.plot(theta, andrews_curve(s, theta), 'r')

# for s in samples[50:70]: # versicolor
#   pl.plot(theta, andrews_curve(s ,theta), 'g')

# for s in samples[100:120]: # virginica
#   pl.plot(theta, andrews_curve(s, theta), 'b')

pl.xlim(-np.pi,np.pi)
pl.show()
