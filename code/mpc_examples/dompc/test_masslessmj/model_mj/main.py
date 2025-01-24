import numpy as np
import mujoco
import glfw
# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

import do_mpc

from controller_dyn import control

# import mujoco interface
from mj_interface import mjmod_init, mjrend_init, linearize, setpolelen

# import opengl to save results as a mp4
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# make function to do co design using finite difference
def balance(delta_t = 0.02, plotting = False, polelen = 0.6, M = 10, m = 5):

    #### stuff for equations comparison
    import do_mpc
    from model_dynamics import model_setdynamics

    model_dyn = model_setdynamics(M,m,polelen)

    mpc = control(model_dyn, delta_t)

    est_dyn = do_mpc.estimator.StateFeedback(model_dyn)
    sim_dyn = do_mpc.simulator.Simulator(model_dyn)
    sim_dyn.set_param(t_step = delta_t)
    sim_dyn.setup()


    x0_dyn = np.array([0, 0, 0, 0])

    mpc.x0 = x0_dyn
    sim_dyn.x0 = x0_dyn
    est_dyn.x0 = x0_dyn
    

    mpc.set_initial_guess()
    u = 0
    ####

    # set initial conditions
    x0 = [0, np.pi]
    model, data = mjmod_init(x0)

    ##model.body_mass[1] = M
    #model.body_mass[2] = m

    

    # set pole length and update mass/intertia
    #setpolelen(model, data, polelen)
    #print(model.body_mass)
    #print(model.body_ipos)
    #print(model.body_inertia)


    # init window
    window, camera, scene, context, viewport = mjrend_init(model, data)
    
    # set matrices for plotting
    xarr = []
    thetaarr = []
    dxarr = []
    dthetaarr = []
    farr = []
    #jarr = []
    tarr = []

    xdyn = []
    thetadyn = []
    dxdyn = []
    dthetadyn = []


    # start main loop
    x = np.zeros(4)
    step = 1
    rgb = []
    depth = []

    xtol = .1
    thetatol = .05 # 3 degrees
    dxtol = 0.1
    dthetatol = .05

    # set up recording
    if plotting:
        width, height = glfw.get_framebuffer_size(window)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('cartpole.mp4', fourcc, 30.0, (width, height))

    stop = False
    Fail = False
    while(stop==False):
        if step != 1:
            if abs(curx) < xtol and abs(curtheta) < thetatol and abs(curdx) < dxtol and abs(curdtheta) < dthetatol: # if success
                stop = True
            elif glfw.window_should_close(window): # if window is closed
                stop = True
            #elif abs(curdx) < dxtol and abs(curtheta) > thetatol and abs(curx) > xtol:
               # stop = True
               # Fail = True


        if plotting:
            # record image from window and save to video:
            glReadBuffer(GL_FRONT)
            pixel_data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
            # Convert pixel data to a numpy array
            image = np.frombuffer(pixel_data, dtype=np.uint8)
            image = image.reshape(height, width, 3)
            # Flip the image vertically (OpenGL's origin is at the bottom left)
            image = np.flipud(image)
            video_writer.write(image)

        u = mpc.make_step(x0_dyn)
        x0_dyn = sim_dyn.make_step(u)

        data.ctrl = u
        curf = u
        curt = delta_t*step

        mujoco.mj_step(model, data)

        xdyn = np.append(xdyn, x0_dyn[0])
        thetadyn = np.append(thetadyn, x0_dyn[2])
        dxdyn.append(x0_dyn[1])
        dthetadyn.append(x0_dyn[3])

        
        curx = data.qpos[0]
        curtheta = data.qpos[1]
        curdx = data.qvel[0]
        curdtheta = data.qvel[1]

        # append arrays
        xarr = np.append(xarr, curx)
        thetaarr = np.append(thetaarr, curtheta)
        dxarr = np.append(dxarr, curdx)
        dthetaarr = np.append(dthetaarr, curdtheta)
        farr = np.append(farr, curf)
        tarr = np.append(tarr, curt)

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
    if plotting:
        video_writer.release()

        # plot timeseries
        import matplotlib.pyplot as plt
        thetaarr = np.pi - thetaarr

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.suptitle('States and Controls Over Entire Range')
        fig.tight_layout()

        # position states
        ax1.plot(tarr, xarr, label='MuJoCo')
        ax1.plot(tarr, xdyn, label='Dynamics')
        
        ax2.plot(tarr, thetaarr, label='MuJoCo')
        ax2.plot(tarr, thetadyn, label='Dynamics')

        ax3.plot(tarr, dxarr, label='MuJoCo')
        ax3.plot(tarr, dxdyn, label='Dynamics')

        ax4.plot(tarr, -dthetaarr, label='MuJoCo')
        ax4.plot(tarr, dthetadyn, label='Dynamics')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        ax1.set_ylabel('X')
        ax2.set_ylabel('Theta')
        ax3.set_ylabel('DX')
        ax4.set_ylabel('DTheta')


        ax3.set_xlabel('Time')
        plt.savefig('timeseris_comp', bbox_inches='tight')


        # make animation
        from animate_cartpole import animate_cartpole
        animate_cartpole(xarr, thetaarr, farr, gif_fps=20, l=polelen, save_gif=True, name='mujoco_sim.gif')

        animate_cartpole(xdyn, thetadyn, farr, gif_fps=20, l=polelen, save_gif=True, name='dynamics_sim.gif')


if __name__ == "__main__":
    balance(plotting=True)
