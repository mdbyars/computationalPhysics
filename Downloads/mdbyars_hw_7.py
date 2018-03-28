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

for i in range(len(y)):
	f = lambda i: ((y[i]*2*np.pi)/(x[i]**2 + x[i])) 
	y[i] = f(i)

#now y is just Cltt 
dy = np.zeros(len(y), np.float)
dy[0:-1] = np.diff(y)/np.diff(x)
dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

plt.plot(dy)
plt.yscale('log')
plt.show()

print(dy)
def riemannSum(N, a, b, f):
	dx = (b-a)/N
	#inerating from a to b
	i = a
	# summing up the areas
	area = 0
	yS = []
	xS = []
	#looping from lower bound (a) to upper bound (b)
	while i <=b:
#		print(i)
		height = f(i)
		area += dx * height*-1
		#yS will be a list of all the areas
		yS = yS + [area]
		xS = xS + [i]
		i += dx
	plt.plot(xS, yS)
	plt.show()
	return area	

are = riemannSum(400, 1, 1999, lambda i: dy[i]*x[i] /(2*np.pi))
print(are)
