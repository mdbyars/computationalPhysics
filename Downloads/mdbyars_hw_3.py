from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import math 

hbar = 1.055*10**(-34)
KB = 1.381*10**(-23)
c = 299792458
def radInt(omeg, T):
#	print(hbar*omeg/(KB*T))
	return (hbar * omeg**3)/(4* math.pi**2* c**2 (math.exp(hbar*omeg/(KB*T))-1))

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
		print(i)
		height = f(i)
		area += dx * height
		#yS will be a list of all the areas
		yS = yS + [area]
		xS = xS + [i]
		i += dx
#	plt.plot(xS, yS)
#	plt.show()
	return area	

def infRiemannSum(N, a, b, f):
	ans = 0
	if type(a) == type(3):
		lower = a
		upper = a+10
		dif = 1 
		while (dif > ans*0.01 or upper<a+100): 
			dif = ans - riemannSum(N, lower, upper, f)
			ans += riemannSum(N, lower, upper, f)
			lower+=upper
			upper = lower+100
		#	print(lower, upper)
	 
	elif type(b) == type(3):
		upper = b
		lower = b-10
		dif = 1 
		while (dif > ans*0.01 or lower>b-100): 
			dif = ans - riemannSum(N, lower, upper, f)
			ans += riemannSum(N, lower, upper, f)
		upper+=-100
		lower = upper - 100
#		print(lower, upper)
	return ans
def wRad(T, y):
#	print(x**3)
#	print(math.exp(x))
	return (KB**4 * T**4)/(4* math.pi**2 * c**2 * hbar**3) * infRiemannSum(1000, 0, "P", (lambda x: x**3 / (2.7182 **x)))    

#	return infRiemannSum(1000, 0, "P", (lambda x: x**3 / (math.exp(x))))    
print(wRad(4, 4))


#5.14
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
        return ans


G = 6.674*10**-11
rho = 10
        #return G*rho*z*integr

F = (lambda z : z*(integrateGaussQuad(100, -5, 5, (lambda y: (integrateGaussQuad(200, -5, 5, (lambda x: 1/((x**2 + y**2 + z**2)**(3/2)))))))))

i = 0
fXs = []
fYs = []
while i <=10:
        fXs += [i]
        print(F(i))
        fYs += [F(i)*G*rho]
        i += 0.1
plt.plot(fXs, fYs)
plt.show()

#5.21
def potential(r, q):
	return q/(4*math.pi*(1.6*10**(-19))*r+0.0001)

def doubleRPotential(a, b, i, j):
	r1 = math.sqrt((i-a[0])**2+(j-a[1])**2)
	r2 = math.sqrt((i-b[0])**2+(j-b[1])**2)
	return potential(r1, 1)+potential(r2, -1)

#point a 
a = [5, 0]
b = [10, 0]
c = np.zeros((20,20))
for i in range(19): 
	for j in range(19): 
		c[i][j]=doubleRPotential(a, b, i, j)
plt.pcolor(c)
plt.show()
