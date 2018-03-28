from scipy.interpolate import interp1d 
import numpy as np 
from io import StringIO 
from sympy import Derivative  
#regexp = r"(\d+\.\d+E\+\d)"
#xcol = np.fromregex('./camb_scalcls.dat')[0]

xcol = np.loadtxt('./camb_scalcls.dat')
ycol = np.loadtxt('./camb_scalcls.dat')
xval = lambda ival: xcol[ival][0]
yval = lambda ival: ycol[ival][1]

y = []
x = []
for i in range(2000):
	y = y+ [yval(i)]
	x = x+ [xval(i)]

yspline = []
xspline = []
i = 0
while i<=2000:
	yspline = yspline+ [yval(i)]
	xspline = xspline+ [xval(i)]
	i+=10
f = interp1d(xspline, yspline)
k=0
yinter = []
while k<2000:
	yinter += [f(x[k])]
	k+=1	

xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt 
plt.plot(x, y,'r--', x, yinter, 'b--')
plt.show()
ydif = []
j=0
while j<2000:
	ydif += [y[j]-yinter[j]]
	j+=1
plt.plot(x, ydif)
plt.show()
f = lambda i : i*2
for i in range(len(y)):
	#y[i] = Derivative(lambda i: (y[i]/(x[i]**2 + x[i])), x[i]) 
	y[i] = Derivative(f, 2) 
plt.plot(x, y)
plt.xscale('log')
plt.show()
