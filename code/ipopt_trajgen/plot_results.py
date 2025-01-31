import numpy as np
import matplotlib.pyplot as plt

def pl_ts(xarr,thetaarr,dx,dtheta,tarr,farr,name='cartpole_traj'):
    fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(5)
    fig.suptitle('States and Controls Over Entire Range')
    fig.tight_layout()

    # position states
    ax1.plot(tarr, xarr)
    ax2.plot(tarr, thetaarr)
    ax3.plot(tarr, dx)
    ax4.plot(tarr, dtheta)
    ax5.plot(tarr, farr)


    ax1.set_ylabel('x1')
    ax2.set_ylabel('dx1')
    ax3.set_ylabel('x2')
    ax4.set_ylabel('dx2')
    ax5.set_ylabel('u')

    ax3.set_xlabel('Time')
    plt.savefig(name, bbox_inches='tight')