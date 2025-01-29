# First party modules
from pyoptsparse import IPOPT, Optimization
import numpy as np

# MPC and system parameters (cartpole)
dt = 0.01
N = 200 # horizon
ns = 4 # number of states
nu = 1 # number of controls
Q = np.eye(ns) # weighting matrix for states
R = np.eye(nu) # weighting matrix for controls
Qf = Q # terminal weighting matrix for states
#Rf = R not sure if this is needed (check later)

# using s for state vector to not confuse with coordinate x
def dynamics(s,u):
    # representation of dynamics: ds = f(s,u)
    # s: {x, theta, dx, dtheta}
    # ds: (dx, dtheta, ddx, ddtheta)

    # cartpole parameters: m: mass of pole, M: mass of cart, L: length of pole
    m = 2.5
    M = 7.5
    L = 1
    g = 9.81

    # get states
    x = s[0]
    theta = s[1]
    dx = s[2]
    dtheta = s[3]

    # calculate ds
    det = m * L * np.cos(theta)**2 - L * (M + m)
    ddx = (-m * L * g * np.sin(theta) * np.cos(theta) - L * (u + m * L * dtheta**2 * np.sin(theta))) / det
    ddtheta = ((M + m) * g * np.sin(theta) + np.cos(theta) * (u + m * L * dtheta**2 * np.sin(theta))) / det
    
    ds = [dx, dtheta, ddx, ddtheta]

    return ds

# function to represent discrete cartpole dynamics using scipy to solve ode
def discretize(s,u,t,dt):
    # takes in current states at time n and current time: s[n] and t[n] as well as dt
    # returns states at next time n+1: s[n+1]
    # use integration: s[n+1] = s[n] + integral from t[n] to t[n+1] of f(s,u)dt
    import scipy.integrate as int # use scipy to solve ode

    # integral (need to check, im confused about this...)
    res = int.quad(lambda t: dynamics(s,u), t, t+dt)
    snext = s + res
    return snext



"""
# rst begin objfunc
def objfunc(xdict, Q, R, Qf):
    
    s = xdict["svars"]
    funcs = {}
    funcs["obj"] = np.transpose(s)
    conval = [0] * 2
    conval[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] - 72.0
    conval[1] = -x[0] - 2.0 * x[1] - 2.0 * x[2]
    funcs["con"] = conval
    fail = False

    return funcs, fail


# rst begin optProb
# Optimization Object
optProb = Optimization("TP037 Constraint Problem", objfunc)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 2, lower=None, upper=0.0)

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
print(optProb)

# rst begin OPT
# Optimizer
optOptions = {}
opt = IPOPT(options=optOptions)

# rst begin solve
# Solve
sol = opt(optProb, sens="FD")

# rst begin check
# Check Solution
print(sol)"""
