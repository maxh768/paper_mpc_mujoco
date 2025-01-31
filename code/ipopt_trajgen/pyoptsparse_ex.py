# First party modules
from pyoptsparse import IPOPT, Optimization
import numpy as np

# MPC and system parameters (cartpole)
dt = 0.5
N = 7 # horizon
ns = 4 # number of states
nu = 1 # number of controls
Q = np.eye(ns) # weighting matrix for states
R = np.eye(nu) # weighting matrix for controls
R[0,0] = 1e-4
Qf = Q # terminal weighting matrix for states
#Rf = R not sure if this is needed (check later)
s0 = np.array([2, -2, 1, 3]) # initial state

# using s for state vector to not confuse with coordinate x
"""
cart pole dynamics and discretization nonlinear, using linear system for now
def dynamics(t,s,u):
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
    from scipy.integrate import solve_ivp # use scipy to solve ode

    # integrate to get s[n+1]
    sol = solve_ivp(dynamics, [t, t+dt], s, args=(u,), t_eval=[t+dt])
    snext = [sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0]]

    return snext

def test_discretization():
    # tests discretization function to see if result is realistic
    x = []
    theta = []
    s0 = np.array([0, 0, 0, 0])
    s = s0
    x.append(s[0])
    theta.append(s[1])
    for i in range(200):
        s = discretize(s,50,0,dt)
        #print(s)
        x.append(s[0])
        theta.append(s[1])
    from animate_cartpole import animate_cartpole
    animate_cartpole(x, theta, np.ones(201), save_gif=True)
"""

def discretize(s,u,t,dt):
    # model of two osciliating masses (discrete linear model from dompc)
    # dt = 0.5
    # s = {x1, v1, x2, v2}
    # u = {f}
    # s[n+1] = A*s[n] + B*u[n]
    A = np.array([[ 0.763,  0.460,  0.115,  0.020],
              [-0.899,  0.763,  0.420,  0.115],
              [ 0.115,  0.020,  0.763,  0.460],
              [ 0.420,  0.115, -0.899,  0.763]])

    B = np.array([[0.014],
                [0.063],
                [0.221],
                [0.367]])
    
    s = np.array([[s[0]],
                [s[1]],
                [s[2]],
                [s[3]]])
    
    u = np.array([[u]])

    snext = A@s + B@u
    return np.transpose(snext)


# rst begin objfunc
def objfunc(vars):
    
    s = vars["s"]
    u = vars["u"]


    funcs = {}

    
    terminal = np.transpose(s[N-4:N]).dot(Qf).dot(s[N-4:N])
    # initialize cost
    funcs["obj"] = terminal
    
    # constraint 1: initial condition
    con_1 = s[0:4] - s0
    

    # constraint 2: dynamics
    con_2 = np.zeros(ns*(N-1))

    # sum cost from n = 0 to N-1
    for i in range(N-1):
        # sum cost
        funcs["obj"] += np.transpose(s[i:i+4]).dot(Q).dot(s[i:i+4]) + np.transpose(u[i])*R*u[i]

        # fill in second constraint
        index1 = (i+1)*ns
        index2 = index1 + 4
        con_2[i:i+4] = s[index1:index2] - discretize(s[i:i+4],u[i],i*dt,dt)
        

    funcs["con_ic"] = con_1
    funcs["con_dyn"] = con_2

    fail = False
    return funcs, fail


# problem
optProb = Optimization("Cart Pole Trajectory Optimization", objfunc)

# design variables: states (s)
optProb.addVarGroup("s", ns*N, "c")

# design variables: controls (u)
optProb.addVarGroup("u", nu*N, "c")

# add constraints
optProb.addConGroup("con_ic", ns, lower=0.0, upper=0.0)
optProb.addConGroup("con_dyn", ns*(N-1), lower=0.0, upper=0.0)

# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
#print(optProb)

# rst begin OPT
# Optimizer
optOptions = {}
opt = IPOPT(options=optOptions)

# rst begin solve
# Solve
sol = opt(optProb, sens="FD")

# rst begin check
# Check Solution
#print(sol)

x = np.zeros(N)
theta = np.zeros(N)
dx = np.zeros(N)
dtheta = np.zeros(N)

for i in range(N):
    cx1 = sol.xStar['s'][i*ns]
    x[i] = cx1

    cx2 = sol.xStar['s'][i*ns+ 1]
    theta[i] = cx2

    cx3 = sol.xStar['s'][i*ns + 2]
    dx[i] = cx3

    cx4 = sol.xStar['s'][i*ns + 3]
    dtheta[i] = cx4

u = sol.xStar['u']
t = np.linspace(0, N*dt, N)
# plot results
from plot_results import pl_ts
pl_ts(x,theta,dx,dtheta,t,u)

