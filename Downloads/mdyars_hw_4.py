import numpy as np
import math
import matplotlib.pyplot as plt
def H(n, x): 
	if n == 0:
		return 1
	if n == 1:
		return 2*x
	else:
		return 2*x*H(n-1, x) - 2*n*H(n-2, x)

def psi(n, x):
	u = math.exp(-x**2/2) / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
	return u* H(n, x)


xs = np.linspace(-4, 4, 100)
ys=[]
for i in range(4):
	for j in range(len(xs)):
		ys += [psi(i, xs[j])]
	plt.plot(xs, ys)
	ys = []
		

plt.show()

xs = np.linspace(-10, 10, 100)
ys = []
for i in range(len(xs)):
	ys += [psi(3, xs[i])]

plt.plot(xs, ys)
plt.show()

#Gauss Quad written by Mark Newman 
from numpy import ones,copy,cos,tan,pi,linspace

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def integrateGaussQuad(n, a, b, f):
        x, w = gaussxwab(n, a, b)
        ans = 0
        for i in range(len(x)):
                ans += w[i]* f(x[i])
	print(ans)
        return ans

def infGauss(N, a, b, f):
	ans = 0
	if type(a) == type(3):
		lower = a
		upper = a+10
		dif = 1 
		while (dif > ans*0.01 or upper<a+100): 
			dif = ans - integrateGaussQuad(N, lower, upper, f)
			print("firstndhalf")
			ans += integrateGaussQuad(N, lower, upper, f)
			lower+=upper
			upper = lower+100
			print(lower, upper)
	 
	elif type(b) == type(3):
		upper = b
		lower = b-10
		dif = 1 
		while (dif > ans*0.01 or lower>b-100): 
			dif = ans - integrateGaussQuad(N, lower, upper, f)
			ans += integrateGaussQuad(N, lower, upper, f)
			upper+=-100
			print("secondhalf")
			lower = upper - 100
	return ans
def QHO(x):
	return x**2 * (psi(5, x)**2)

ansnw = math.sqrt(infGauss(50, 0, "P", QHO)+infGauss(50, "N", 0, QHO))
print(ansnw)
	
