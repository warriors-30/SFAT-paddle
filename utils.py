import copy
import paddle
from paddle.vision import datasets,transforms
from paddle.io import Dataset
from sampling import cifar_iid, svhn_iid, cifar_noniid_skew, svhn_noniid_skew, cifar100_noniid_skew, svhn_noniid_unequal, cifar10_noniid_unequal
import os
import numpy as np
from PIL import Image

class custom_cifar10(Dataset):
    def __init__(self, root, mode, transform=None):
        super(custom_cifar10, self).__init__()
        self.targets=[]
        self.root=root
        self.mode=mode
        self.transform = transform
        data_path=os.path.join(self.root,'cifar-10-python.tar.gz')
        self.dataset=datasets.Cifar10(data_path, mode=self.mode,transform=self.transform)
        for i in range(len(self.dataset)):
            self.targets.append(self.dataset[i][1])

    def __getitem__(self, index):
        image,_ = self.dataset[index]
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.dataset)

class custom_cifar100(Dataset):
    def __init__(self, root, mode, transform=None):
        super(custom_cifar100, self).__init__()
        self.targets = []
        self.root = root
        self.mode = mode
        self.transform = transform
        data_path = os.path.join(self.root, 'cifar-100-python.tar.gz')
        self.dataset = datasets.Cifar100(data_path, mode=self.mode, transform=self.transform)
        for i in range(len(self.dataset)):
            self.targets.append(self.dataset[i][1])

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.dataset)

class custom_svhn(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform= None):
        super(custom_svhn,self).__init__()

        self.root=root
        self.split=split
        self.transform = transform
        self.target_transform = target_transform

        import scipy.io as sio
        if split=='train':
            self.filename="train_32x32.mat"
        elif split=='test':
            self.filename="test_32x32.mat"
        elif split=='extra':
            self.filename="extra_32x32.mat"
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar-10':
        data_dir = '../data/cifar-10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = custom_cifar10(data_dir, mode='train',transform=apply_transform)

        test_dataset = custom_cifar10(data_dir, mode='test',transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = cifar10_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'svhn':
        data_dir = '../data/svhn/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = custom_svhn(data_dir, split='train',transform=apply_transform)

        test_dataset = custom_svhn(data_dir, split='test', transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = svhn_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = svhn_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                #print(train_dataset.labels)
                user_groups = svhn_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'cifar-100':
        data_dir = '../data/cifar-100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = custom_cifar100(data_dir, mode='train',transform=apply_transform)

        test_dataset = custom_cifar100(data_dir, mode='test',transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar100_noniid_skew(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups

# FedAvg
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = w_avg[key]/len(w)
    return w_avg

# # FedAvg unequal
def average_weights_unequal(w, idx_num):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = w_avg[key]/len(w)
    return w_avg
    
# SFAT
def average_weights_alpha(w, lw, idx, p):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = w_avg[key]/(cou+(len(w)-cou)*p)
    return w_avg


# # SFAT unequal
def average_weights_alpha_unequal(w, lw, idx, p, idx_num):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p * float(idx_num[0]*len(w)/sum(idx_num))
        else:
            w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p * float(idx_num[i]*len(w)/sum(idx_num))
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = w_avg[key]/(cou+(len(w)-cou)*p)
    return w_avg  

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
