import argparse
from data_laoder import TrainKitchenDataset, TestKitchenDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tensorboardX import SummaryWriter
from IPython import embed
from mjrl.utils.fc_network import FCNetwork
from termcolor import colored

SEED = 500

# fix the seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def main(args):
    # log files
    log_dir = '{}{:04d}'.format(args.log_dir, args.exp_id)
    writer = SummaryWriter(log_dir=log_dir)
    # write hyper params to file
    args_dict = vars(args)
    arg_file = open(log_dir + '/args.txt', 'w')
    for arg_key in args_dict.keys():
        arg_file.write(arg_key + " = {}\n".format(args_dict[arg_key]))
    arg_file.close()

    # dataset
    train_dataset = TrainKitchenDataset(path=args.data_path, delta=args.use_delta, normalize=args.normalize,
                                        traj_len=args.traj_len)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = TestKitchenDataset(path=args.data_path, delta=args.use_delta, normalize=args.normalize,
                                      traj_len=args.traj_len)
    test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # define policy
    policy = FCNetwork(obs_dim=2 * args.observation_dim,
                       act_dim=args.action_dim,
                       hidden_sizes=(32,32))
    loss_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(policy.parameters()), lr=args.lr)

    #
    iter_num = 0
    best_test_loss = 1e10
    for e in range(args.num_epoch):
        policy.train()
        # training loop
        for j, data in enumerate(train_dataset_loader):
            state, gt_act = data['state'], data['act']
            optimizer.zero_grad()
            pred_act = policy(state)
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

        # TODO: Need to figure this out
        '''
        params_after_opt = bc_agent.policy.get_param_values()
        bc_agent.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        '''
        # testing loop
        policy.eval()
        test_loss_list = []
        with torch.no_grad():
            for j, data in enumerate(test_dataset_loader):
                state, gt_act = data['state'], data['act']
                pred_act = policy(state)
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
