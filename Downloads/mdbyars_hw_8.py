#segement of code taken from class notes
from math import sin,pi,cos
from numpy import array,arange,sqrt
import matplotlib.pyplot as plt
g=9.81
l=0.1
om0=sqrt(g/l)
f=om0/(2.e0*pi)
T=1./f
#print(T)

def f(r,t):
    theta = r[0]
    omega = r[1]
    ftheta = omega
    fomega = -(g/l)*sin(theta)
    return array([ftheta,fomega],float)

a = 0.0
b = T*20
N = 400
h = (b-a)/N

tpoints = arange(a,b,h)
thetapoints = []
omegapoints = []

r = array([3.0,0.0],float)
for t in tpoints:
    thetapoints.append(r[0])
    omegapoints.append(r[1])
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
plt.plot(tpoints,thetapoints)
#plt.plot(tpoints,omegapoints)
plt.xlabel("t")
plt.xlim([0,20*T])
plt.ylim([-4,4])
#plt.ion()
plt.show()
#end notes segment of code

l = 0.1
g = 9.81
C = 2
U = 0
while U < 0.1: 
	om0=sqrt(g/l)
	f=om0/(2.e0*pi)
	T=1./f
	#print(T)
	def f2(r, t):
		theta = r[0]
		omega = r[1]
		ftheta = omega
		fomega = -(g/l)*sin(theta)+(C*cos(theta)*sin(U*t))
		return array([ftheta,fomega],float)
	
	a = 0.0
	b = T*20
	N = 400
	h = (b-a)/N
	
	tpoints = arange(a,b,h)
	thetapoints = []
	omegapoints = []
	
	r = array([3.0,0.0],float)
	for t in tpoints:
	    thetapoints.append(r[0])
	    omegapoints.append(r[1])
	    k1 = h*f2(r,t)
	    k2 = h*f2(r+0.5*k1,t+0.5*h)
	    k3 = h*f2(r+0.5*k2,t+0.5*h)
	    k4 = h*f2(r+k3,t+h)
	    r += (k1+2*k2+2*k3+k4)/6
	plt.plot(tpoints,thetapoints)
	#plt.plot(tpoints,omegapoints)
	plt.xlabel("t")
	plt.xlim([-U*U,U*U])
	plt.ylim([-U*U,U*U])
	#plt.ion()
	plt.show()
	U+=.5
rho=10.0
rx=28.0
b = 8/3
#om0=sqrt(g/l)
#f=om0/(2.e0*pi)
#T=1./f
#print(T)

def f(ini,t):
    x = ini[0]
    y = ini[1]
    z = ini[2]
    fx = rho*(y-x)
    fy = rx*x-y-x*z
    fz = x*y - b*z
    print(fx,fy,fz)
    return array([fx,fy,fz],float)

a = 0.0
b = 50
N = 400
h = (b-a)/N

tpoints = arange(a,b,h)
xpoints = []
ypoints = []
zpoints = []

r = array([3.0,0.0,0.0],float)
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    zpoints.append(r[2])
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
plt.plot(zpoints,xpoints)
#plt.plot(tpoints,omegapoints)
plt.xlabel("t")
plt.xlim([0,20*T])
plt.ylim([-4,4])
#plt.ion()
plt.show()
#end notes segment of code
