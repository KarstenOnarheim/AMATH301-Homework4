# This is a template for the coding portion of Homework 4 in AMATH 301, 
# Winter 2023 

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

############## Problem 1 ################
## Part a
# Define x(t) below
x = lambda t: 11*(np.exp(-t/12) - np.exp(-t))/6

# Define x'(t) below. I did this one for you.
xprime = lambda t: 11*(-1/12*np.exp(-t/12) + np.exp(-t))/6
A1 = xprime(1.5)

# Example: We use scipy.optimize.fsolve to find zeros of a function. For instance, if

max = scipy.optimize.fsolve(xprime, 1.5)

A2 = max[0]
A3 = x(max[0])


## Part b
# Look at some examples of how we have used fminbound in class, for example on
# January 23, 24, or 27.
negX = lambda t: -1 * x(t)
tval = scipy.optimize.fminbound(negX, 0, 10)
A4 = [tval, x(tval)]

############## Problem 2 ################
## Part a
fxy = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
f = lambda p: fxy(p[0], p[1])

A5 = f([3, 4])
## Part b
A6 = scipy.optimize.fmin(f, [-3, -2])

## Part c

gradf_xy = lambda x,y: np.array([4*x**3 - 42*x + 4*x*y + 2*y**2 - 14,
                                 4*y**3 - 26*y + 4*x*y + 2*x**2 - 22])
gradf = lambda p: gradf_xy(p[0], p[1])

A7 = gradf(A6)
A8 = np.linalg.norm(A7)

## Part d

def gradDes(p):
    tol = 10**-7
    for k in range(2000):
        grad = gradf(p)
        if np.linalg.norm(grad)<tol:
            break
        
        phi = lambda t: p - t*grad
        f_of_phi = lambda t: f(phi(t)) # Create a function of "heights along path"
        tmin = scipy.optimize.fminbound(f_of_phi,0,1) # Find time it takes to reach min height
        p = phi(tmin); # Find the point on the path and update your guess
    return [p, k]

p = [-3, -2]
ans = gradDes(p)
A9 = ans[0]
A10 = ans[1]

## Part e
# Done above!
