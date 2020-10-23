import glob
import os
import pickle as pkl
import numpy as np
import torch
from IPython import embed
from PIL import Image
from torch.utils.data import Dataset


class KitchenDatasetImg(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, traj_len, pkl_path, img_path,
                delta=False, include_vel=True, normalize=False, img_transform=None):
        """[summary]

        Args:
            traj_len ([type]): [description]
            pkl_path ([type]): [description]
            img_path ([type]): [description]
            delta (bool, optional): [description]. Defaults to False.
            include_vel (bool, optional): [description]. Defaults to True.
            normalize (bool, optional): [description]. Defaults to False.
            img_transform ([type], optional): [description]. Defaults to None.
        """
        self.ctrl = []
        self.qpos = []
        self.qvel = []
        self.complete_qpos = []
        self.complete_qvel = []
        self.include_vel = include_vel
        self.delta = delta
        self.normalize = normalize
        self.traj_len = traj_len
        self.img_root_path = img_path
        self.img_path = []
        self.img_transform = img_transform
        for i, r in enumerate(glob.glob(os.path.join(pkl_path, '*.pkl'))):
            with open(r, 'rb') as f:
                data = pkl.load(f)
                self.ctrl.append(data['ctrl'])
                self.qpos.append(data['qpos'][:, :9])       # only include robot state
                self.qvel.append(data['qvel'][:, :9])       # only include robot velocity
                self.complete_qpos.append(data['qpos'])       # only include robot state
                self.complete_qvel.append(data['qvel'])
                if normalize:
                    if delta:
                        a = data['ctrl'] - data['qpos'][:, :9]
                    else:
                        a = data['ctrl']
                    if i == 0:
                        self.stat = a
                    else:
                        self.stat = np.concatenate((self.stat,a))

                # add image path
                img_list = glob.glob(os.path.join(img_path, r.split('/')[-1][:-4], '*.jpg'))
                img_list.sort()
                self.img_path.append(img_list)

        if self.normalize:
            self.mean = self.stat.mean(axis=0)
            self.std = self.stat.std(axis=0)

    def len(self):
        return len(self.ctrl)

    def unnormalize(self, state, act):
        if self.normalize:
            act *= self.std
            act += self.mean

        if self.delta:
            act += state[:9]

        return act

    def getitem(self, item):
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
        if self.img_transform is None:
            start_img = Image.open(self.img_path[item][t]).convert('RGB')
            goal_img= Image.open(self.img_path[item][t + T]).convert('RGB')
        else:    
            start_img = self.img_transform(Image.open(self.img_path[item][t]).convert('RGB'))
            goal_img= self.img_transform(Image.open(self.img_path[item][t + T]).convert('RGB'))
        state_t = torch.from_numpy(state[t].astype(np.float32))
        act = torch.from_numpy(act[t].astype(np.float32))
        
        return {
            'start_img': start_img,
            'goal_img': goal_img,
            'state': state_t,
            'act': act,
            'comp_qpos': torch.from_numpy(self.complete_qpos[item][t].astype(np.float32)),
            'comp_qvel': torch.from_numpy(self.complete_qvel[item][t].astype(np.float32))}


class TrainKitchenDataset(KitchenDatasetImg):
    def __init__(self, traj_len, pkl_path, img_path,
                 delta=False, include_vel=True, normalize=False, img_transform=None):
        KitchenDatasetImg.__init__(self, traj_len, pkl_path, img_path, delta, include_vel, normalize, img_transform)

    def __len__(self):
        return 500

    def __getitem__(self, item):
        return self.getitem(item)


class TestKitchenDataset(KitchenDatasetImg):
    def __init__(self, traj_len, pkl_path, img_path,
                 delta=False, include_vel=True, normalize=False, img_transform=None):
        KitchenDatasetImg.__init__(self, traj_len, pkl_path, img_path, delta, include_vel, normalize, img_transform)

    def __len__(self):
        return self.len()- 500

    def __getitem__(self, item):
        return self.getitem(item+500)