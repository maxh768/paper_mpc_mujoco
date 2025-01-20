import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

"""
## CONFIG SYSTEM
"""

# set number of steps
num_steps = 300
delta_t = 0.01

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)


#set states
x1 = model.set_variable(var_type='_x', var_name='x1', shape=(1,1))
x2 = model.set_variable(var_type='_x', var_name='x2', shape=(1,1))

dx1 = model.set_variable(var_type='_x', var_name='dx1', shape=(1,1))
dx2 = model.set_variable(var_type='_x', var_name='dx2', shape=(1,1))



# control / inputs
tau = model.set_variable(var_type='_u', var_name='tau', shape=(1,1))


# set params
a = 0.5
b = 0.5
mh = 10
m = 5
phi = 0.05
l = a + b
g = 9.81

# dynamics
H22 = (mh + m)*(l**2) + m*a**2
H12 = -m*l*b*np.cos(x2 - x1)
H11 = m*b**2
h = -m*l*b*np.sin(x1-x2)
G2 = -(mh*l + m*a + m*l)*g*np.sin(x2)
G1 = m*b*g*np.sin(x1)

K = 1 / (H11*H22 - (H12**2)) # inverse constant
dx1set = (H12*K*h*dx1**2) + (H22*K*h*dx2**2) - H22*K*G1 + H12*K*G2 - (H22 + H12)*K*tau
dx2set = (-H11*K*h*dx1**2) - (H12*K*h*dx2**2) + H12*K*G1 - H11*K*G2 + ((H12 + H11)*K*tau)


# set rhs
model.set_rhs('x1',dx1)
model.set_rhs('x2',dx2)
model.set_rhs('dx1',dx1set)
model.set_rhs('dx2',dx2set)


model.setup()

"""
## CONFIG CONTROLLER
"""

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 300,
    't_step': delta_t,
    'n_robust': 1,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': False,
    'supress_ipopt_output': True
}
mpc.settings.supress_ipopt_output()
mpc.set_param(**setup_mpc)



# obj function
mterm = (x1-0.19602)**2 + (x2+0.29602)**2 + 0.2*(dx1+2.10662)**2 + 0.2*(dx2+1.4195)**2
lterm = (x1-0.19602)**2 + (x2+0.29602)**2 + 0.2*(dx1+2.10662)**2 + 0.2*(dx2+1.4195)**2
mpc.set_objective(mterm=mterm, lterm=lterm)

# set r term ??
mpc.set_rterm(
    tau=10
)

# lower and upper bounds on states
mpc.bounds['lower','_x','x1'] = -1
mpc.bounds['lower','_x','x2'] = -1
mpc.bounds['upper','_x','x1'] = 1
mpc.bounds['upper','_x','x2'] = 1

# lower and upper bounds on inputs (tau/desired pos?)
mpc.bounds['lower','_u','tau'] = -20
mpc.bounds['upper','_u','tau'] = 20


mpc.setup()

"""
## CONFIG SIMULATOR
"""

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = delta_t)

# uncertain vars (future)

simulator.setup()

"""
## CONTROL LOOP
"""

# initial guess
x0 = np.array([-0.3, 0.2038, -0.41215, -1.0501]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()



# delete all previous results so the animation works
import os 
# Specify the directory containing the files to be deleted 
directory = './results/' 
# Get a list of all files in the directory 
files = os.listdir(directory) 
# Loop through the files and delete each one 
for file in files: 
    file_path = os.path.join(directory, file) 
    os.remove(file_path) 

phibound = [0, 0]
# main loop
from calc_transition import calc_trans
u0 = mpc.make_step(x0)
x0 = simulator.make_step(u0)
#print(mpc.x0['x1',0])
curx1 = mpc.x0['x1',0]
curx2 = mpc.x0['x2',0]
curx3 = mpc.x0['dx1',0]
curx4 = mpc.x0['dx2',0]
numiter = 1

x1arr = []
x2arr = []
tarr = []
farr = []
x3arr = []
x4arr = []
for i in range(num_steps-1):
    curx1 = mpc.x0['x1',0]
    curx2 = mpc.x0['x2',0]
    curx3 = mpc.x0['dx1',0]
    curx4 = mpc.x0['dx2',0]
    cur_t = i*delta_t
    curf = mpc.u0['tau',0]
    #if i % 10 == 0:
    x1arr.append(float(curx1))
    x2arr.append(float(curx2))
    x3arr.append(float(curx3))
    x4arr.append(float(curx4))
    tarr.append(cur_t)
    farr.append(float(curf))
    phibound[0] = phibound[1]
    phibound[1] = curx1 + curx2
    #print('x1: ',curx1)
    #print('x2: ',curx2)
    print('x1 (deg): ', curx1*(180/np.pi))
    print('x2 (deg): ', curx2*(180/np.pi))
    print('x1+x2: ', phibound[1])
    print('step num: ', i+2)
    if (((phibound[0] > -0.1) and (phibound[1] < -0.1)) or ((phibound[0] <-0.1) and (phibound[1] > -0.1))) and curx1>0:
        print('TRANSITION')
        newstates = calc_trans(curx1, curx2, curx3, curx4, m=m, mh=mh, a=a, b=b)
        x0 = np.array([newstates[0], newstates[1], newstates[2], newstates[3]]).reshape(-1,1)
        simulator.x0 = x0
        numiter = numiter + 1
        numpoints = i+2


    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    
twoleg_dir = '/home/max/workspace/paper_mpc_mujoco/vis/twoleg_graphs'

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('States and Controls Over Entire Range')
fig.tight_layout()

# position states
ax1.plot(tarr, x1arr,label='1')
ax1.plot(tarr, x2arr,label='2')

ax2.plot(tarr, x3arr,label='1')
ax2.plot(tarr, x4arr,label='2')

ax3.plot(tarr, farr)

ax1.legend()
ax2.legend()


ax1.set_ylabel('state')
ax2.set_ylabel('state velocity')
ax3.set_ylabel('tau')


plt.savefig('twoleg_states', bbox_inches='tight')


# animate motion of the compass gait
from animate import animate_compass
animate_compass(x1arr, x2arr, a, b, phi, iter=1, saveFig=True, gif_fps=20,name=twoleg_dir + 'twoleg_compass.gif')

