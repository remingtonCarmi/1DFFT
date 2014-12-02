#!/usr/local/bin/python
from math import *
import cmath
import matplotlib.pyplot as plt
import os
import errno
import numpy as np

## 1DFFT.py
## Remi Carmigniani
##This code is an example of how to use fft to calculate function derivatives for periodic functions
##we first calculate the fft of a set with N = 2*q+1 odd
# dx = 2*pi/N
# grid is : x_i = i*dx i=0,..,N-1
#Note that we don't give the x_N point where the function will have the same value than at the x_0 point. 
# calculate u_n = f(x_n) for n = 0 , ... , N-1
# apply the DFT (using the FFT => NLog(N)) this gives U_k for k = -q,...,q (or N terms)
#  calculate the derivative in the spectral domain :  U'_k = sqrt(-1)*k*U_k
# Here we need to be extra careful since the fft in python and pretty much any other languages is set up for k = 0 to N-1 
# so for m = 0 to q 
# just do U'_m = sqrt(-1)*m*U_m
# but for m=q+1 to N-1
# 	U'_m = sqrt(-1)*(m-N)*U_m
# in other words m = k for k = 0 ... q
# and 		 m = k-N for k = q+1 ... N
# this is implemented in the function DFFT(utild,order) for all orders



## Test function
def test(x):
    return 1./(3.+0.5*cos(x)*cos(6*x))
## Chech derivatives
def testp(x):
    return 0.5*(sin(x)*cos(6.*x)+6.*cos(x)*sin(6.*x))*test(x)**2
def test2p(x):
    return (55.5 + 53.5*cos(2*x)+ 150.*cos(5.*x)+294.*cos(7*x) -6.25*cos(10.*x)-16.5*cos(12.*x) - 12.25*cos(14.*x))/(6.+1.*cos(x)*cos(6.*x))**3
#Trap Integration (Copied from 1DDiff.py)
def trapInt(f,dx,nx):
	error = 0.0
	for i in range(0,nx-1):
		error = error+f[i]+f[i+1]	
	return error*.5*dx

def realPart(x):
	return [x[i].real for i in range(len(x))]

def FFT(u):
	return np.fft.fft(u) 
def InvFFT(utild):
	#n=len(utild)-1
	#return [sum([1./float(n+1)*utild[i]*cmath.exp(1j*2*pi*i*k/float(n+1)) for i in range(n/2)])+sum([1./float(n+1)*utild[i]*cmath.exp(1j*2*pi*(i-n-1)*k/float(n+1)) for i in range(n/2,n+1)]) for k in range(n+1)]
	return np.fft.ifft(utild)

def power(a,b):
	if abs(a*a)>0:
		return cmath.exp(b*cmath.log(a))
	else:
		return 0
def DFFT(utild,order): 
	n = len(utild)
	result = [0 for i in range(n)]
	for i in range(n/2):
		result[i] = utild[i]*power(1j*i,order)
	for i in range(n/2,n):
		result[i] = utild[i]*power(1j*(i-n),order)
	return result
		
	
	
## Discretization parameter 
N=511
L=2*pi
dx=L/float(N)

x = [dx*ii for ii in range(N)] 

u = [test(x[ii]) for ii in range(N)] 

#Show Original data
#plt.plot(x, u)
#plt.ylabel('U')
#plt.xlabel('x')
#plt.axis([0 , L , -1.1, 1.1]) 
#plt.show()

#FFT
utild = FFT(u)
#utild = np.append(utild,0)


#Organization of the data :
ureconst = realPart(InvFFT(utild))
error=sqrt(trapInt([(u[i]-ureconst[i])**2 for i in range(N)],dx,N))
print 'Error on the reconstruction ' + repr(error)

#Calculate the derivative
uptild = DFFT(utild,2.)
up = realPart(InvFFT(uptild))
error = sqrt(trapInt([(up[i]-test2p(x[i]))**2 for i in range(N)],dx,N))
print 'Error on the reconstruction ' + repr(error)

#plt.plot(np.absolute(utild))
#plt.show()
plt.plot(x,ureconst)
plt.plot(x,u)
plt.plot(x,up)
plt.plot(x,[test2p(x[i]) for i in range(N)])
plt.show()



