# helpful for visualizing what policy is doing
import argparse
from data_laoder import TrainKitchenDataset, TestKitchenDataset
import torch
import numpy as np
from mjrl.utils.fc_network import FCNetwork
import adept_envs
import gym
import time
import os
import matplotlib.pyplot as plt
from IPython import embed
from util import try_cv2_import
cv2 = try_cv2_import()
import io
SEED = 500

# fix the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#np.random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def main(args):
    test_dataset = TestKitchenDataset(path=args.data_path, delta=args.use_delta, normalize=args.normalize,
                                      traj_len=args.traj_len)

    # define policy
    policy = FCNetwork(obs_dim=2 * args.observation_dim,
                       act_dim=args.action_dim,
                       hidden_sizes=(32, 32))

    # laod model
    policy.load_state_dict(torch.load(args.model_path))
    policy.eval()

    #
    env = gym.make('kitchen_relax-v1')

    if args.store_vis:
        fig = plt.figure(figsize=(20, 10))
    #
    count = 0
    with torch.no_grad():
        while True:
            for data in test_dataset:
                count += 1
                state, gt_act = data['state'], data['act']
                goal_state = state[args.observation_dim:].numpy()
                if args.random_goal_state:
                    indx = np.random.randint(len(test_dataset.ctrl))
                    qpos, qvel = test_dataset.qpos[indx], test_dataset.qvel[indx]
                    indx = np.random.randint(len(qpos))
                    goal_state = np.concatenate((qpos[indx], qvel[indx]))

                if args.store_vis:
                    log_dir = os.path.join(args.log_dir, "{:04d}".format(count))
                    if not os.path.isdir(log_dir):
                        os.makedirs(log_dir)
                    env.reset()
                    env.sim.data.qpos[:] = goal_state[:30]
                    env.sim.data.qvel[:] = goal_state[30:args.observation_dim]
                    env.sim.forward()
                    goal_img = env.render(mode='rgb_array')

                start_state = state[:args.observation_dim].numpy()
                env.reset()
                # set the initial state
                act_mid = env.act_mid
                act_rng = env.act_amp
                env.sim.data.qpos[:] = start_state[:30]
                env.sim.data.qvel[:] = start_state[30:args.observation_dim]
                env.sim.forward()
                for i in range(args.num_steps):
                    state = np.concatenate((env.sim.data.qpos, env.sim.data.qvel))
                    inp = torch.from_numpy(np.concatenate((state, goal_state)).astype(np.float32))

                    pred_act = policy(inp).detach().numpy()
                    if args.use_delta or args.normalize:
                        pred_act = test_dataset.unnormalize(state, pred_act)

                    # execute action in real world
                    ctrl = (pred_act - env.sim.data.qpos[:9]) / (env.skip * env.model.opt.timestep)
                    act = (ctrl - act_mid) / act_rng
                    act = np.clip(act, -0.999, 0.999)
                    next_obs, reward, done, env_info = env.step(act)

                    if args.render:
                        env.render()
                    print("num_steps = {}".format(i))

                    if args.store_vis:
                        cur_img = env.render(mode='rgb_array')
                        plt.subplot(121)
                        plt.title('Cur Image')
                        plt.imshow(cur_img)
                        plt.xticks([])
                        plt.yticks([])

                        plt.subplot(122)
                        plt.title('Gaol Image')
                        plt.imshow(goal_img)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for visualizing behaviour cloning')
    # for kitchen dataset
    parser.add_argument('--use_delta', action='store_true', default=False,
                        help='whether to predict delta or not')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize the data or not')
    parser.add_argument('--data_path', type=str,
                        default='/home/dhiraj/Downloads/kitchen_demos_multitask_extracted_data',
                        help='place where data pickle files are stored')
    parser.add_argument('--observation_dim', type=int, default=59)
    parser.add_argument('--traj_len', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=9)

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