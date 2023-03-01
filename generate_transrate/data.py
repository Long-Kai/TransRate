from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
import torchvision.transforms as transforms
import numpy as np
import torch

def load_transfrom(name='cifar100', source='imagenet', mode='train'):
    _name = name.lower()

    if source == 'cifar10':
        im_size = 32
    else:
        re_size = 256
        im_size = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if mode == 'train':
        if _name == "cifar10" or _name=="cifar100":
            transformer = transforms.Compose([transforms.Resize(im_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        elif _name == "imagenet":
            transformer = transforms.Compose([transforms.Resize(re_size),
                                              transforms.RandomResizedCrop(im_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        else:
            transformer = transforms.Compose([
                transforms.Resize(re_size),
                transforms.RandomResizedCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    elif mode == 'test':
        if _name == "cifar10" or _name=="cifar100":
            transformer = transforms.Compose([transforms.Resize(im_size),
                                              transforms.ToTensor(),
                                              normalize])
        elif _name == "imagenet":
            transformer = transforms.Compose([transforms.Resize(re_size),
                                              transforms.CenterCrop(im_size),
                                              transforms.ToTensor(),
                                              normalize])
        else:
            transformer = transforms.Compose([
                transforms.Resize(re_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor()
            ])
    return transformer


def load_orig_dataset(name, source, dataset_path="./datasets", train=True, target_transform=None, download=True):
    mode = 'test'  # use only test transform to avoid randomness in training data
    transformer = load_transfrom(name=name, source=source, mode=mode)
    if name == "cifar100":
        ds = CIFAR100(root=dataset_path, train=train, transform=transformer,
                      target_transform=target_transform, download=download)
    elif name == "cifar10":
        ds = CIFAR10(root=dataset_path, train=train, transform=transformer,
                     target_transform=target_transform, download=download)
    elif name == "imagenet":
        ds = ImageNet(root=dataset_path, train=train, transform=transformer,
                      target_transform=target_transform, download=download)

    return ds


def get_num_class(name):
    if name == "cifar100":
        num_class = 100
    elif name == "cifar10":
        num_class = 10
    elif name == "imagenet":
        num_class = 1000

    return num_class


class TLdataset:
    def __init__(self, name, args):
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.num_workers = args.num_dataprovider_workers
        self.args = args
        self.source = args.source

        data_rng = np.random.RandomState(seed=args.data_seed)
        self.data_seed = data_rng.randint(1, 999999)

        self.name = name.lower()
        self.num_class = get_num_class(self.name)


        dataset_tr = load_orig_dataset(name=name, source=self.source, train=True)
        dataset_te = load_orig_dataset(name=name, source=self.source, train=False)

        self.loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=args.batch_size,
                                                         shuffle=True,
                                                         num_workers=args.num_dataprovider_workers)
        self.loader_te = torch.utils.data.DataLoader(dataset_te, batch_size=50,
                                                         shuffle=False,
                                                         num_workers=args.num_dataprovider_workers)









