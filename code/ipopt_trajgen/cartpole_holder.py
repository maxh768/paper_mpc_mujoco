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
        x.append(s[0])
        theta.append(s[1])
    from animate_cartpole import animate_cartpole
    animate_cartpole(x, theta, np.ones(201), save_gif=True)
"""