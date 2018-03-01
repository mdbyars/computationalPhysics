
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import math 
from decimal import Decimal


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

		
#
#def fZi(z, y):
#	#inner integral
#	i = 0
#	integr = 0
#	gaus = gaussxwab(100, -5, 5)
#	while i < len(gaus[0]):
#		integr += gaus[1][i]*(1/(gaus[0][i]**2 * y**2 * z**2)**(3/2))
#		i+=1
#	return integr
#def fZo(z):
#	#outter integral 
#	i = 0 
#	integr = 0
#	gaus = gaussxwab(100, -5, 5)
#	while i < len(gaus[0]):
#		integr += gaus[1][i]*fZi(z, gaus[0][i])
#		i+= 1
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

