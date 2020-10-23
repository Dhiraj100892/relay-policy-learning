# load pickel file and render images
from IPython import embed

import adept_envs
import gym
import cv2
import glob
import pickle as pkl
import argparse
import os
from pyvirtualdisplay import Display


def extract_data(inp_path, out_path):

    # setup env
    display_ = Display(visible=0, size=(550, 500))
    display_.start()

    env = gym.make('kitchen_relax-v1')
    env.reset()

    # load pickle files
    pkl_files = glob.glob(os.path.join(inp_path, '*.pkl'))
    pkl_files.sort()

    for p in pkl_files:
        print("opening pkl file  = {}".format(p))
        # mkdir
        path = os.path.join(out_path, p.split('/')[-1][:-4])
        if not os.path.isdir(path):
            os.makedirs(path)

        # load datd
        with open(p, 'rb') as f:
            data = pkl.load(f)

        # render images
        for i_frame in range(data['ctrl'].shape[0]):
            env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
            env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
            env.sim.forward()
            img = env.render('rgb_array')
            cv2.imwrite(os.path.join(path,'{:05d}.jpg'.format(i_frame)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for data extarction')
    parser.add_argument('--path', type=str, help="path to the data dir", default ='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data')
    parser.add_argument('--out-path', type=str, help="path to the save data", default='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/imgs')
    args = parser.parse_args()
    extract_data(args.path, args.out_path)
    
