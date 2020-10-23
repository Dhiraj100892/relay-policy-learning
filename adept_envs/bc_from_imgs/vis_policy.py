# for visualizing policy

import argparse
import torch
import os
import sys
import time
import cv2
import io
import glob
import adept_envs
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from data_loader import TrainKitchenDataset, TestKitchenDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from IPython import embed
from termcolor import colored
from model import FCNetwork
from PIL import Image
from pyvirtualdisplay import Display
import gym


SEED = 500

# fix the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#np.random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


def main(args):
    
    # define the device ===========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # define trasnform
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)])

    restore_transform = transforms.Compose([
        DeNormalize(*mean_std),
        transforms.ToPILImage()])

    # define dataset
    test_dataset = TestKitchenDataset(traj_len=args.traj_len, pkl_path=args.pkl_path, img_path=args.img_path,
                                        delta=args.use_delta, normalize=args.normalize,
                                        img_transform=img_transform)
    
    # define model
    vision_model = models.resnet18(pretrained=True)
    vision_model.fc = Identity()
    vision_model = vision_model.to(device)
    vision_model.eval()

    policy = FCNetwork(512 + 512 + 18, 9, (256, 128, 64))
    policy.load_state_dict(torch.load(args.model_path))
    policy = policy.to(device)
    policy.eval()

    # start display 
    display_ = Display(visible=0, size=(550, 500))
    display_.start()

    #
    env = gym.make('kitchen_relax-v1')

    if args.store_vis:
        fig = plt.figure(figsize=(20, 10))
    #
    count = 0

    # Evaluation loop
    policy.eval()
    with torch.no_grad():
        while True:
            for data in test_dataset:
                count += 1
                start_img = data['start_img'].to(device)
                goal_img = data['goal_img'].to(device)
                robot_state = data['state'].to(device)
                gt_act = data['act'].to(device)
                comp_qpos = data['comp_qpos'].numpy()
                comp_qvel = data['comp_qvel'].numpy()
                
                # TODO: implement 
                if args.random_goal_state:
                    pass

                # to do start img goal img render back for cv2
                if args.store_vis:
                    log_dir = os.path.join(args.log_dir, "{:04d}".format(count))
                    if not os.path.isdir(log_dir):
                        os.makedirs(log_dir)
                    goal_img_pil = restore_transform(data['goal_img'])

                # set env
                env.reset()
                act_mid = env.act_mid
                act_rng = env.act_amp
                env.sim.data.qpos[:] = comp_qpos
                env.sim.data.qvel[:] = comp_qvel
                env.sim.forward()

                goal_img_enc = vision_model(torch.unsqueeze(goal_img, 0))
                
                for i in range(args.num_steps):
                    start_img_cv2 = env.render(mode='rgb_array')
                    start_img_pil = Image.fromarray(start_img_cv2)
                    start_img_tensor = img_transform(start_img_pil).to(device)

                    # generate policy inp
                    start_img_enc = vision_model(torch.unsqueeze(start_img_tensor, 0)) 

                    # concatenate the input
                    robot_state = torch.from_numpy(
                        np.concatenate((env.sim.data.qpos[:9], env.sim.data.qvel[:9])).astype(np.float32)).to(device)
                    robot_state = torch.unsqueeze(robot_state, 0)

                    inp = torch.cat((start_img_enc, goal_img_enc, robot_state), -1)
                    pred_act = policy(inp)

                    if args.use_delta or args.normalize:
                        pred_act = test_dataset.unnormalize(env.sim.data.qpos, pred_act.detach().cpu().numpy())

                    # execute action in real world
                    ctrl = (pred_act.reshape(-1) - env.sim.data.qpos[:9]) / (env.skip * env.model.opt.timestep)
                    act = (ctrl - act_mid) / act_rng
                    act = np.clip(act, -0.999, 0.999)
                    next_obs, reward, done, env_info = env.step(act)

                    print("count = {} step = {}".format(count, i))

                    if args.store_vis:
                        cur_img = env.render(mode='rgb_array')
                        plt.subplot(121)
                        plt.title('Cur Image')
                        plt.imshow(cur_img)
                        plt.xticks([])
                        plt.yticks([])

                        plt.subplot(122)
                        plt.title('Goal Image')
                        plt.imshow(np.array(goal_img_pil)[:, :, ::-1])
                        plt.xticks([])
                        plt.yticks([])

                        #img = get_img_from_fig(fig)
                        fig.canvas.draw()
                        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                        # img is rgb, convert to opencv's default bgr
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(log_dir, "{:03d}.jpg".format(i)), img)
                        plt.clf()
                
                # convert gif 
                os.system('ffmpeg -framerate 6 -i {}/%03d.jpg {}/out.gif'.format(log_dir, log_dir))

                # rm all the jpg images
                img_list = glob.glob('{}/*.jpg'.format(log_dir))
                for img in img_list:
                    os.system('rm {}'.format(img))
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for behaviour cloning')
    # for kitchen dataset
    parser.add_argument('--use_delta', action='store_true', default=True,
                        help='whether to predict delta or not')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='whether to normalize the data or not')
    parser.add_argument('--pkl_path', type=str,
                        default='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data',
                        help='place where data pickle files are stored')
    parser.add_argument('--img_path', type=str,
                        default='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/imgs',
                        help='place where data pickle files are stored')
    parser.add_argument('--traj_len', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--test_iter', type=int, default=100)

    # for evaluating
    parser.add_argument('--num_steps', type=int, default=15)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--random_goal_state', action='store_true', default=False,
                        help='whether to send zero goal state to policy')
    parser.add_argument('--store_vis', action='store_true', default=False,
                        help='whether to predict delta or not')
    parser.add_argument('--render', action='store_true', default=False,
                        help='whether to predict delta or not')
    parser.add_argument('--log_dir', default='./vis_logs',
                        help='directory to save agent logs (default: ./logs/)')
    args = parser.parse_args()
    main(args)