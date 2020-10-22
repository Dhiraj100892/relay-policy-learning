import matplotlib.pyplot as plt
import argparse
import glob
import pickle as pkl
import os
import numpy as np
from IPython import embed


def vis(path, num_points):
    """

    """
    ctrl_data = None
    pos_data = None
    for p in glob.glob(os.path.join(path, "*.pkl")):
        with open(p, "rb") as f:
            data = pkl.load(f)
            if ctrl_data is None:
                ctrl_data = data['ctrl']
                pos_data = data['qpos']
            else:
                ctrl_data = np.concatenate((ctrl_data, data['ctrl']))
                pos_data = np.concatenate((pos_data, data['qpos']))

    plt.subplot(211)
    plt.plot(range(num_points), ctrl_data[:num_points])

    plt.subplot(212)
    plt.plot(range(num_points), pos_data[:num_points, :9])

    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Arguments for data visualization')
    parser.add_argument('--path', type=str, help="path to the data dir")
    parser.add_argument('--num_points', type=int, default=100, help="number of points to visualize")
    args = parser.parse_args()
    vis(args.path, args.num_points)