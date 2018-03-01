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
#		print(i)
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
		while (dif > ans*0.01 or upper<700): 
			dif = ans - riemannSum(N, lower, upper, f)
			ans += riemannSum(N, lower, upper, f)
			lower+=100
			upper = lower+100
			print(lower, upper)
	 
	elif type(b) == type(3):
		upper = b
		lower = b-10
		dif = 1 
		while (dif > ans*0.01 or lower>-700): 
			dif = ans - riemannSum(N, lower, upper, f)
			ans += riemannSum(N, lower, upper, f)
		upper+=-100
		lower = upper - 100
#		print(lower, upper)
	return ans
def wRad(T, x):
	print(math.exp(x))
	return (KB**4 * T**4)/(4* math.pi**2 * c**2 * hbar**3) * infRiemannSum(1000, 0, "P", (lambda x: x**3 / ((2.71828182845 **x))))  


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


def fZi(z, y):
	#inner integral
	i = 0
	integr = 0
	gaus = gaussxwab(100, -5, 5)
	while i < len(gaus[0]):
		integr += gaus[1][i]*(1/(gaus[0][i]**2 * y**2 * z**2)**(3/2))
	return integr
def fZo(z):
	#outter integral 
	i = 0 
	integr = 0
	gaus = gaussxwab(100, -5, 5)
	while i < len(gaus[0]):
		integr += gaus[1][i]*fZi(z, gaus[0][i])
	G = 6.674*10**-11
	rho = 10
	return G*rho*z*integr

i = 0
fXs = []
fYs = []
while i <=10: 
	fXs += [i]
	fYs += [fZo(i)]
	i += 0.1
plt.plot(fXs, fYs)
plt.show()
