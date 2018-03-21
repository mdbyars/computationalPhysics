from math import exp
from copy import deepcopy
import matplotlib.pyplot as plt
x = 1.0 
a = 1.0
b = 2.0
for x  in range(10):
	y = b/(a+x**2)
	fxt = -x + a*y +x**2 * y
	print("diverging points")
	print(fxt)

th = 0.0 
last = 1.0
i = 0
ans = []
xs = [last]
while(i< 300):
        th = deepcopy((b + last)/(2*a +2*last**2))
	samesig = str(th)[0:4] == str(last)[0:4]
        if samesig:
		j = 0
                while samesig and j < 200:
                        ans = ans + [th]
                        xs += [th]
			last = deepcopy(th)
        		th = deepcopy((b + last)/(2*a +2*last**2))
                        j += 1
			samesig = str(th)[0:4] == str(last)[0:4]
			if( j == 199):
				i = 300
				print("ANSWER")
				print(th/2)
		i+=1
        else:
                i+=1
		last = deepcopy(th)



def mySecant(f, x0, x1, n):
	y0 = f(x0)
	y1 = f(x1)
	for i in range(1, n):
		x = x1 - (x1-x0)*y1/(y1-y0)
		y = f(x)
		x0 = x1
		y0 = y1
		x1 = x
		y1 = y

	return x


def fr(r):
	G = 6.674 * 10**-11
	M = 5.974 * 10**24
	m = 7.348 * 10**22
	R = 3.844 * 10**8 
	w = 2.662 * 10**-6
	return ((G*M)/(r**2))-((G*m)/((R-r)**2))-(w**2 * r)

print mySecant(fr, 100, 1000, 30)



