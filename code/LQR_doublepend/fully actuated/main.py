import numpy as np
import mujoco
import glfw


# import mujoco interface
from mj_interface import mjmod_init, mjrend_init, linearize

# import opengl to save results as a mp4
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

"""
Balanced fully actuated double pendulum using basic LQR control
 - need to figure out how to do better LQR control (optimize force used, total time, add constraints to problem)
 - underactuated system is simulated in other module
 - need to figure out how to do path planning (collocation?)
 - need to convert to MPC
"""


# set initial conditions and dt
delta_t = 0.02
x0 = [0, 0]
model, data = mjmod_init(x0)

###***********LQR*********###

# linearize system for LQR controller (at target state)
lin_state = np.array([0, 0, 0, 0]) # linearize at target state  (balanced)
lin_u = np.array([0, 0]) # target control = 0
# make sure to check A and B matrices are correct later
A,B = linearize(model, data, lin_state, lin_u)

# define Q and R matrices
Q = np.eye(4)
R = np.eye(2)

# define feedback matrices (assume full feedback)
C = np.eye(4)
D = 0*np.eye(2)

# import and instantiate LQR controller
from LQR import LQR
LQR = LQR(A, B, C, D, delta_t)

# calculate feedback gains
LQR.calculate_K_lqr(Q, R)

#calculate reference feedback gains (not entirely sure what this is...)
#LQR.calculate_K_r()

###***********LQR*********###


###**********simulation setup*********###
# init window
window, camera, scene, context, viewport = mjrend_init(model, data)

# start main loop
step = 1
rgb = []
depth = []


# set up recording
width, height = glfw.get_framebuffer_size(window)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('doub_pend.mp4', fourcc, 30.0, (width, height))

# init arrays
tarr = [] # time
u1arr = [] # input 1
u2arr = [] # input 2

# arrays from mj
th1arr = []
th2arr = []
th1darr = []
th2darr = []

stop = False
###**********simulation setup*********###


# set initial condition for simulation
IC = np.array([np.pi, 0, 0, 0])
data.qpos = IC[:2]
data.qvel = IC[2:]

# run main loop
while(stop==False):

    if step != 1:
        if glfw.window_should_close(window): # if window is closed
            stop = True

    # record image from window and save to video:
    glReadBuffer(GL_FRONT)
    pixel_data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
    # Convert pixel data to a numpy array
    image = np.frombuffer(pixel_data, dtype=np.uint8)
    image = image.reshape(height, width, 3)
    # Flip the image vertically (OpenGL's origin is at the bottom left)
    image = np.flipud(image)
    video_writer.write(image)

    # mj step 1: pre control
    mujoco.mj_step1(model, data)

    ###***********CTRL*********###

    # get current state of system
    state = np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])

    # calculate continous control from LQR feedback
    u = LQR.feedback(state, np.array([0, 0])) # assume reference is 0

    # calculate discrete control from LQR feedback
    u_d = LQR.feedback_d(state, np.array([0, 0])) # assume reference is 0

    print('CONTINOUS CTRL:', u)
    print('DISCRETE CTRL:', u_d)

    # use continous control, need to figure out what the difference is in this
    data.ctrl = u


    ###***********CTRL*********###

    # mj step2: run with ctrl input
    mujoco.mj_step2(model, data)

    th1arr.append(data.qpos[0])
    th2arr.append(data.qpos[1])
    th1darr.append(data.qvel[0])
    th2darr.append(data.qvel[1])
    curt = delta_t*step
    tarr.append(curt)
    u1arr.append(u[0])
    u2arr.append(u[1])

    step += 1
    # render frames

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)


    glfw.swap_buffers(window)
    glfw.poll_events()

# close window
glfw.terminate()

# save video
video_writer.release()

# plot timeseries
from plot_results import pl_ts
pl_ts(th1arr, th2arr, th1darr, th2darr, u1arr, u2arr, tarr, name='doubpend-fullyactuated.pdf')

