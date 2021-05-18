import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import pandas as pd
import ipdb

def replay_traj():
    Xr = pd.read_csv("MPC_Xr7.csv", header=None).to_numpy()
    Xh = pd.read_csv("MPC_Xh7.csv", header=None).to_numpy()
    goals = pd.read_csv("MPC_goal8.csv", header=None).to_numpy()
    msee_X = pd.read_csv("MPC_msee7.csv", header=None).to_numpy()
    # Xr = pd.read_csv("SEA_Xr.csv", header=None).to_numpy()
    # Xh = pd.read_csv("SEA_Xh.csv", header=None).to_numpy()
    # msee_X = pd.read_csv("SEA_msee.csv", header=None).to_numpy()
    msee_X = msee_X.reshape((Xh.shape[0], Xh.shape[0], msee_X.shape[1]), order='F')

    ax = plt.subplot(111, aspect='equal')
    dmin = 1.5
    # for i in range(Xh.shape[1]-1):
    for i in range(200):
        ax.cla()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        xH = Xh[:,i]
        xR = Xr[:,i]
        msee_x = msee_X[:,:,i]
        g = goals[:,i]

        # ax.scatter([g[0]], [g[1]], c='k', s=250)
        ax.scatter([xH[0]], [xH[1]], c='b', s=250)
        ax.scatter([xR[0]], [xR[1]], c='r', s=250)

        cov = msee_x[:,[0, 1]][[0, 1],:]
        lambda_, v = np.linalg.eig(cov)

        n_std = 3
        angle = np.rad2deg(np.arccos(v[0, 0] / np.linalg.norm(v[:,0])))
        # 3 std ellipse
        ell = Ellipse(xy=(xH[0],xH[1]),
                      width=lambda_[0]*n_std*2, height=lambda_[0]*n_std*2,
                      angle=angle, fill=None, color="red")
        ax.add_patch(ell)

        # safety margin
        ell = Ellipse(xy=(xH[0],xH[1]),
                      width=(dmin*2-0.6), height=(dmin*2-0.6),
                      fill=None)
        ax.add_patch(ell)

        plt.pause(0.01)
        # ipdb.set_trace()
        # input(": ")
        ax.patches = []

    print("done")
    plt.show()

def overlay_traj(ax):
    Xr = pd.read_csv("../data/ecos_X.csv", header=None).to_numpy()
    Xh = pd.read_csv("../data/ecos_X_hat.csv", header=None).to_numpy()
    msee_X = pd.read_csv("../data/ecos_msee.csv", header=None).to_numpy()

    # Xr = pd.read_csv("../data/cosmo_X_active.csv", header=None).to_numpy()
    # Xh = pd.read_csv("../data/cosmo_X_hat_active.csv", header=None).to_numpy()
    # msee_X = pd.read_csv("../data/cosmo_msee_active.csv", header=None).to_numpy()

    msee_X = msee_X.reshape((Xh.shape[0], Xh.shape[0], msee_X.shape[1]), order='F')

    ax.set_xlim(-12, 10)
    ax.set_ylim(-10, 10)
    idxs = [2, 50, 100, 140, 199]
    human_pos = Xh[:,idxs]
    robot_pos = Xr[:,idxs]
    msee_eig = [np.linalg.eig(msee_X[:,:,i][:,[0, 1]][[0, 1],:]) for i in idxs]

    blues = cm.get_cmap('Blues', 5)

    for k, i in enumerate(idxs):
        lambda_, v = msee_eig[k]
        xh = human_pos[:,k]
        n_std = 3
        angle = np.rad2deg(np.arccos(v[0, 0] / np.linalg.norm(v[:,0])))

        if k == 0:
            ell = Ellipse(xy=(xh[0],xh[1]),
                          width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                          angle=angle, fill=True, color=blues(min(i+50, 199) / 199), hatch="/", alpha=0.6)
            ax.add_patch(ell)

    ax.plot(Xr[0,:], Xr[1,:], c="xkcd:dark red", lw=0.5)
    ax.plot(Xh[0,:], Xh[1,:], c="xkcd:dark blue", lw=0.5)

    ax.scatter(human_pos[0,:], human_pos[1,:], c=idxs, cmap="Blues", vmin=-50, s=200)
    ax.scatter(robot_pos[0,:], robot_pos[1,:], c=idxs, cmap="Oranges", vmin=-50, s=200)


if __name__ == "__main__":
    ax = plt.subplot(111, aspect="equal")
    overlay_traj(ax)
    plt.show()