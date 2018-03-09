import numpy as np 
import math 
a = np.array([[4, -1, -1, -1],[-1, 3, 0, -1],[-1, -1, -1, 4], [-1, 0, 3, -1]])
b = np.array([1, 0, 0, 1])
print(np.linalg.solve(a,b))

#start quantum section 

L = 5
a = 10
M = 9.1094*10**(-31) 
hba = 1.0545*10**(-31)

def vi(a, L):
	return a*(L**2)/(2*L) 
def hamil(M, f):
	return 0

def Hmn(L, m, n):
	if m == n: 
		return ((-hba*2)/(2.0*M))*((-3.141592*n)/(L))*(2.0/L)*((L**2.0/4.0))/3
	elif m%2 == n%2:
		return 0
	else:
		c = 2.0*L/3.14159265
		d = (m*n)/(((m**2.0)-(n**2.0))**2)
		return ((-hba**2)/(2*M))*((3.141592*n)/(L))*(-2/L)*((((c**2)*d))+a)
A = np.zeros((10, 10))

for m in range(9): 
	for n in range(9): 
		A[m][n] = Hmn(L, m+1, n+1)


print(np.linalg.eigvals(A))

#print(np.linalg.eigvalsh(A))
#
B = np.zeros((100, 100))
#
for z in range(99):
	for k in range(99):
		B[z][k] = Hmn(L, z+1, k+1)

print(np.linalg.eigvals(B))
