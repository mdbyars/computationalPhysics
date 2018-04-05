
from math import sin,pi,cos
from numpy import array,arange,sqrt
import matplotlib.pyplot as plt
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
    print(x,y,z) 
    fx = rho*(y-x)
    print(rx)
    fy = rx*x-y-x*z
    fz = x*y - b*z
    #print(fx,fy,fz)
    return array([fx,fy,fz],float)

a = 0.0
b = 5
N = 10
h = (b-a)/N

tpoints = arange(a,b,h)
xpoints = []
ypoints = []
zpoints = []

r = array([3.0,2.0,1.0],float)
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    zpoints.append(r[2])
    k1 = h*f(r,t)
    print(k1)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
plt.plot(zpoints,xpoints)
#plt.plot(tpoints,omegapoints)
plt.xlabel("t")
plt.xlim([0,200])
plt.ylim([-4,4])
#plt.ion()
plt.show()
#end notes segment of code
