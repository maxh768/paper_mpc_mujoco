# First party modules
from pyoptsparse import IPOPT, Optimization
import numpy as np

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
    
    s = np.array([[float(s[0])],
                [float(s[1])],
                [float(s[2])],
                [float(s[3])]])
    
    u = np.array([[u]])
    s_calc = A@s + B@u
    snext = [float(s_calc[0]), float(s_calc[1]), float(s_calc[2]), float(s_calc[3])]

    return snext

def test_discretization():
    # tests discretization function to see if result is realistic
    x = []
    theta = []
    dx = []
    dtheta = []
    t = []
    u = []
    s0 = np.array([2, -2, 1, 3]) # initial state
    s = s0
    x.append(s[0])
    theta.append(s[1])
    dx.append(s[2])
    dtheta.append(s[3])
    t.append(0)
    u.append(0)
    for i in range(200):
        s = discretize(s,0.05,0,dt)
        x.append(float(s[0]))
        theta.append(float(s[1]))
        dx.append(float(s[2]))
        dtheta.append(float(s[3]))
        t.append((i+1)*dt)
        u.append(0)
        # plot results
    from plot_results import pl_ts
    pl_ts(x,theta,dx,dtheta,t,u)

class trajgen:
    def __init__(self,
                 dt,
                 N,
                 Q,
                 Qf,
                 R,
                 ns,
                 nu,
                 s0):
        self.dt = dt
        self.N = N
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.ns = ns
        self.nu = nu
        self.s0 = s0

    def objfunc(self,vars):
        N = self.N
        ns = self.ns
        nu = self.nu
        Q = self.Q
        Qf = self.Qf
        R = self.R

        # objective and constraints for trajectory optimization (direct transcription)
        s = vars["s"]
        u = vars["u"]

        funcs = {}

        terminal = np.transpose(s[N-4:N]).dot(Qf).dot(s[N-4:N])
        # initialize cost
        funcs["obj"] = terminal
        
        # constraint 1: initial condition
        #print(s0)
        con_1 = s[0:4] - s0

        # constraint 2: dynamics
        con_2 = np.zeros(ns*(N-1))
        con_3 = np.zeros(nu*N)

        # sum cost from n = 0 to N-1
        for i in range(N-1):
            # sum cost
            funcs["obj"] += np.transpose(s[i:i+4]).dot(Q).dot(s[i:i+4]) + np.transpose(u[i])*R*u[i]

            # fill in second constraint
            index1 = (i+1)*ns
            index2 = index1 + 4

            con_2[ns*i:ns*i+4] = s[index1:index2] - discretize(s[ns*i:ns*i+4],u[i],i*dt,dt)
            con_3[i] = u[i]
            
        funcs["con_ic"] = con_1
        funcs["con_dyn"] = con_2
        funcs["con_u"] = con_3

        fail = False
        return funcs, fail
    def traj_opt(self):
        N = self.N
        ns = self.ns
        nu = self.nu
        dt = self.dt

        # problem
        optProb = Optimization("Cart Pole Trajectory Optimization", self.objfunc)

        # design variables: states (s)
        optProb.addVarGroup("s", ns*N, "c")

        # design variables: controls (u)
        optProb.addVarGroup("u", nu*N, "c", value=0)

        # add constraints
        optProb.addConGroup("con_ic", ns, lower=0.0, upper=0.0)
        optProb.addConGroup("con_dyn", ns*(N-1), lower=0.0, upper=0.0)
        optProb.addConGroup("con_u", nu*(N), lower=-1, upper=1)

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
        u = sol.xStar['u']
        return u[0]
        """x = np.zeros(N)
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
        pl_ts(x,theta,dx,dtheta,t,u)"""

if __name__ == "__main__":
    # MPC and system parameters
    dt = 0.5
    N = 7 # horizon
    ns = 4 # number of states
    nu = 1 # number of controls
    Q = np.eye(ns) # weighting matrix for states
    R = np.eye(nu) # weighting matrix for controls
    #R[0,0] = 1
    Qf = Q # terminal weighting matrix for states
    #Rf = R not sure if this is needed (check later)
    s0 = np.array([2, -2, 1, 3]) # initial state
    # using s for state vector to not confuse with coordinate x

    x1 = []
    dx1 = []
    x2 = []
    dx2 = []
    t = []
    u = []
    u0 = 0

    # run MPC loop
    for i in range(100):
        x1.append(s0[0])
        dx1.append(s0[1])
        x2.append(s0[2])
        dx2.append(s0[3])
        t.append(i*dt)
        u.append(u0)

        p = trajgen(dt,N,Q,Qf,R,ns,nu,s0)
        u0 = p.traj_opt()
        s0 = discretize(s0,u0,0,dt)

    from plot_results import pl_ts
    pl_ts(x1,dx1,x2,dx2,t,u)


