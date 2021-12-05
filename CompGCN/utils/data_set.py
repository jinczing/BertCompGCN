from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        np.random.randint(2)
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)
        return triple, label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

class TrainBinaryDataset(Dataset):
    def __init__(self, pos_triplets, neg_triplets, num_ent, params):
        super(TrainBinaryDataset, self).__init__()
        self.p = params
        self.pos_triplets = pos_triplets
        self.neg_triplets = neg_triplets
        self.triplets = self.pos_triplets+self.neg_triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        # print(item, ele, ele['label'])
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        hard_label = label.clone()
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)
        return triple, label, hard_label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        # print(self.num_ent, label)
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.from_numpy(y)

class BinarySampler(Sampler):

    def __init__(self, pos_num, neg_num, length_before_new_iter=500):
        super(BinarySampler, self).__init__([])
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.length_before_new_iter = length_before_new_iter

    def __len__(self):
        return self.length_before_new_iter

    def __iter__(self):
        l = []
        for i in range(self.length_before_new_iter):
            l.append(np.random.randint(self.pos_num))
            # ran = np.random.randint(2)
            # if ran:
            #     l.append(np.random.randint(self.pos_num))
            # else:
            #     l.append(self.pos_num+np.random.randint(self.neg_num))
        return iter(l)

class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

class TestBinaryDataset(Dataset):
    def __init__(self, pos_triplets, neg_triplets, num_ent, params):
        super(TestBinaryDataset, self).__init__()
        self.pos_triplets = pos_triplets
        self.neg_triplets = neg_triplets
        self.triplets = self.pos_triplets + self.neg_triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        hards = label.clone()
        return triple, label, hards

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

class InferenceDataset(Dataset):
    def __init__(self):
        super().__init__()
        pass
