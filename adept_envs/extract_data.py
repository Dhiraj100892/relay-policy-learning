import os
import glob
import sys
import argparse
from termcolor import colored

def extract_data(path):
    """
    path to the data folder
    sttructure data_folder/exp/exp_log.mjl
    """
    sys.path.append('/home/dhiraj/project/puppet/vive/source')
    for p in os.listdir(path):
        if '.' in p:
            continue
        cmd = 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so && python adept_envs/utils/parse_demos.py ' \
              '--env kitchen_relax-v1 ' \
              '--demo_dir ' + os.path.join(path, p) + '/  --view playback --skip 40 --render offscreen'
        print(colored(cmd, color='red'))
        os.system(cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for data extarction')
    parser.add_argument('--path', type=str, help="path to the data dir")
    args = parser.parse_args()
    extract_data(args.path)