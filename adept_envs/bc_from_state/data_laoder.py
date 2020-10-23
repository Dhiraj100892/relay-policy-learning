from torch.utils.data import Dataset
import glob
import os
import pickle as pkl
import numpy as np
import torch
from IPython import embed


class KitchenDataset(Dataset):
    def __init__(self, traj_len, path='/home/dhiraj/Downloads/kitchen_demos_multitask_extracted_data',
                 delta=False, include_vel=True, normalize=False):
        self.ctrl = []
        self.qpos = []
        self.qvel = []
        self.include_vel = include_vel
        self.delta = delta
        self.normalize = normalize
        self.traj_len = traj_len
        for i, r in enumerate(glob.glob(os.path.join(path, '*.pkl'))):
            with open(r, 'rb') as f:
                data = pkl.load(f)
                self.ctrl.append(data['ctrl'])
                self.qpos.append(data['qpos'])
                self.qvel.append(data['qvel'])
                if normalize:
                    if delta:
                        a = data['ctrl'] - data['qpos'][:, :9]
                    else:
                        a = data['ctrl']
                    if i == 0:
                        self.stat = a
                    else:
                        self.stat = np.concatenate((self.stat,a))

        if self.normalize:
            self.mean = self.stat.mean(axis=0)
            self.std = self.stat.std(axis=0)

    def _len(self):
        return len(self.ctrl)

    def unnormalize(self, state, act):
        if self.normalize:
            act *= self.std
            act += self.mean

        if self.delta:
            act += state[:9]

        return act

    def _getitem(self, item):
        state = self.qpos[item]
        if self.include_vel:
            state = np.concatenate((state, self.qvel[item]), axis=1)

        act = self.ctrl[item]
        if self.delta:
            act -= state[:, :9]

        if self.normalize:
            act -= self.mean
            act /= self.std

        T = np.random.randint(1, self.traj_len)
        t = np.random.randint(0, len(state)-T)
        return {'state': torch.from_numpy(np.concatenate((state[t], state[t+T])).astype(np.float32)),
                'act': torch.from_numpy(act[t].astype(np.float32))}


class TrainKitchenDataset(KitchenDataset):
    def __init__(self, traj_len, path='/home/dhiraj/Downloads/kitchen_demos_multitask_extracted_data',
                 delta=False, include_vel=True, normalize=False):
        KitchenDataset.__init__(self, traj_len, path, delta, include_vel, normalize)

    def __len__(self):
        return 500

    def __getitem__(self, item):
        return self._getitem(item)


class TestKitchenDataset(KitchenDataset):
    def __init__(self, traj_len, path='/home/dhiraj/Downloads/kitchen_demos_multitask_extracted_data',
                 delta=False, include_vel=True, normalize=False):
        KitchenDataset.__init__(self, traj_len, path, delta, include_vel, normalize)

    def __len__(self):
        return self._len() - 500

    def __getitem__(self, item):
        return self._getitem(item + 500)


