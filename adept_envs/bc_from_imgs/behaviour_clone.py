
import argparse
import torch
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
import time


SEED = 500

# fix the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def main(args):
    
    # define the device ===========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log files
    log_dir = '{}{:04d}'.format(args.log_dir, args.exp_id)
    writer = SummaryWriter(log_dir=log_dir)
    
    # write hyper params to file
    args_dict = vars(args)
    arg_file = open(log_dir + '/args.txt', 'w')
    for arg_key in args_dict.keys():
        arg_file.write(arg_key + " = {}\n".format(args_dict[arg_key]))
    arg_file.close()

    # define trasnform
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    normal_mean_std = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)])

    # define dataset
    train_dataset = TrainKitchenDataset(traj_len=args.traj_len, pkl_path=args.pkl_path, img_path=args.img_path,
                                        delta=args.use_delta, normalize=args.normalize,
                                        img_transform=img_transform)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    test_dataset = TestKitchenDataset(traj_len=args.traj_len, pkl_path=args.pkl_path, img_path=args.img_path,
                                        delta=args.use_delta, normalize=args.normalize,
                                        img_transform=img_transform)
    test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # define model
    vision_model = models.resnet18(pretrained=True)
    vision_model.fc = Identity()
    vision_model = vision_model.to(device)
    vision_model.eval()

    policy = FCNetwork(512 + 512 + 18, 9, (256, 128, 64))
    policy = policy.to(device)

    # define optimizer & loss
    loss_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(policy.parameters()), lr=args.lr)

    # training loop
    iter_num = 0
    best_test_loss = 1e10
    for e in range(args.num_epoch):
        policy.train()
        # training loop
        for j, data in enumerate(train_dataset_loader):
            start_img = data['start_img'].to(device)
            goal_img = data['goal_img'].to(device)
            robot_state = data['state'].to(device)
            gt_act = data['act'].to(device)

            optimizer.zero_grad()

            # generate policy inp
            with torch.no_grad():
                start_img_enc = vision_model(start_img) 
                goal_img_enc = vision_model(goal_img)

            # concatenate the input
            inp = torch.cat((start_img_enc, goal_img_enc, robot_state), -1)
            pred_act = policy(inp)
            loss = loss_criterion(pred_act, gt_act)
            loss.backward()
            optimizer.step()

            # log the values, basically loss
            iter_num += 1
            writer.add_scalar('train/loss', loss.item(), iter_num)
            print("Train_loss = {:.3f} iter = {:06d}".format(loss.item(), iter_num))
            # model saving code as well
            if iter_num % args.save_iter == 0:
                torch.save(policy.state_dict(),
                           log_dir + '/model_{}.pth'.format(iter_num))

        # testing loop
        if iter_num % args.test_iter == 0:

            policy.eval()
            test_loss_list = []
            with torch.no_grad():
                for j, data in enumerate(test_dataset_loader):
                    start_img = data['start_img'].to(device)
                    goal_img = data['goal_img'].to(device)
                    robot_state = data['state'].to(device)
                    gt_act = data['act'].to(device)
                    

                    # generate policy inp
                    start_img_enc = vision_model(start_img) 
                    goal_img_enc = vision_model(goal_img)

                    # concatenate the input
                    inp = torch.cat((start_img_enc, goal_img_enc, robot_state), -1)
                    pred_act = policy(inp)
                    loss = loss_criterion(pred_act, gt_act)
                    test_loss_list.append(loss.item())

            test_loss = np.array(test_loss_list).mean()
            writer.add_scalar('test/loss', test_loss, iter_num)
            print(colored("Test_loss = {:.3f} epoch = {:06d}".format(test_loss, e), 'red'))
            
            # save model
            if test_loss < best_test_loss:
                torch.save(policy.state_dict(),
                        log_dir + '/best_loss.pth'.format(iter_num))
                best_test_loss = test_loss


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

    # for training
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--log-dir', default='./logs/',
                        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument('--exp-id', type=int, required=True, help='name for storing the logs')
    args = parser.parse_args()
    main(args)