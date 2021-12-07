from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import random
from ordered_set import OrderedSet

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
        # triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        # label = self.get_label(label)
        # hard_label = label.clone()
        # if self.label_smooth != 0.0:
        #     label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)
        # return triple, label, hard_label

        subj, obj, label = torch.tensor(ele[0]), torch.tensor(ele[1]), torch.tensor(ele[2])
        return subj, obj, label, item

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.from_numpy(y)

class BinarySampler(Sampler):

    def __init__(self, pos_num, neg_num, asym_num=None, hards=None, length_before_new_iter=50000):
        super(BinarySampler, self).__init__([])
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.hards = hards
        self.length_before_new_iter = length_before_new_iter

    def __len__(self):
        return self.length_before_new_iter

    def __iter__(self):
        l = []
        # for i in range(self.length_before_new_iter):
        #     l.append(np.random.randint(self.pos_num+self.neg_num))
        # print(self.pos_num, self.neg_num, flush=True)
        p = np.ones((self.pos_num+self.neg_num))
        p[self.hards] *= 10
        p /= p.sum()
        # for i in range(self.length_before_new_iter):
            # if self.hards is not None and np.random.randint(10)==0 and len(self.hards):
            #     l.append(self.hards[np.random.randint(len(self.hards))])
            # else:

            # ran = np.random.randint(self.pos_num+self.neg_num)
        l = np.random.choice(self.pos_num+self.neg_num, (self.length_before_new_iter)//2, p=p).tolist()
        ll = []
        for i in l:
            if i%2==0:
                ll.append(i)
                ll.append(i+1)
            else:
                ll.append(i)
                ll.append(i-1)
        
        # if ran<self.pos_num:
        # if ran%2 == 0:
        #     l.append(ran+1)
        # else:
        #     l.append(ran-1)

            # ran = np.random.randint(2)
            # if not ran:
            #     l.append(np.random.randint(self.pos_num))
            # else:
            #     l.append(self.pos_num+np.random.randint(self.neg_num))
        return iter(l)

class TestBinarySampler(Sampler):

    def __init__(self, pos_num, neg_num, length_before_new_iter=None):
        super(TestBinarySampler, self).__init__([])
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.length_before_new_iter = length_before_new_iter

    def __len__(self):
        return self.length_before_new_iter

    def __iter__(self):
        l = []
        for i in range((self.pos_num+self.neg_num)//2):
            l.append(i)
        random.shuffle(l)
        l = OrderedSet(l)
        ll = []
        while(len(l)):
            t = l.pop()
            ll.append(t*2)
            ll.append(t*2+1)

        # for i in range(self.length_before_new_iter):
        #     l.append(np.random.randint(self.pos_num))
        # print(self.pos_num, self.neg_num, flush=True)
            # ran = np.random.randint(2)
            # if ran:
            #     l.append(np.random.randint(self.pos_num))
            # else:
            #     l.append(self.pos_num+np.random.randint(self.neg_num))
        return iter(ll)

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
        # triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        # label = self.get_label(label)
        # hard_label = label.clone()
        # return triple, label, hard_label

        subj, obj, label = torch.tensor(ele[0]), torch.tensor(ele[1]), torch.tensor(ele[2])
        return subj, obj, label, item

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
