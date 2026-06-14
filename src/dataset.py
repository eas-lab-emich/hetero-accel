import logging
import os.path
import shutil
import numpy as np
import torch.utils.data
from crimson_magick.cifar_zoo import get_test_loader, Cifar
from torchvision import transforms
import torchvision.datasets as datasets
from src.datasets.imagenet_dataset import ImagenetDataset

logger = logging.getLogger(__name__)


def load_data(dataset, dataset_path, arch, batch_size, workers,
              validation_split, train_size, valid_size, test_size, test_only,
              verbose, to_cpu):
    loader_kwargs = {
        'batch_size': 64,
        'num_workers': 16,
        'pin_memory': False if to_cpu else torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 4,
        'drop_last': False
    }
    if "cifar" in dataset:
        return None, None, get_test_loader(Cifar[dataset.upper()], loader_kwargs)

    dataset_fn = __dataset_factory(dataset, batch_size=batch_size)
    train_loader, valid_loader, test_loader = get_data_loaders(dataset_fn, dataset_path, arch, batch_size, workers,
                                                               validation_split, train_size, valid_size, test_size,
                                                               test_only)
    if verbose:
        if test_only:
            logger.info('Dataset sizes:\n\ttest={}'.format(len(test_loader.sampler)))
        else:
            logger.info('Dataset sizes:\n\ttraining={}\n\tvalidation={}\n\ttest={}'.format(
                    len(train_loader.sampler) if train_loader is not None else 0,
                    len(valid_loader.sampler) if valid_loader is not None else 0,
                    len(test_loader.sampler)
                )
            )
    return train_loader, valid_loader, test_loader


def __dataset_factory(dataset, batch_size):
    try:
        return {
            # image classification
            'cifar10': get_cifar10_dataset,
            'cifar100': get_cifar100_dataset,
            'imagenet': get_imagenet_dataset,
            'tiny-imagenet': get_tinyimagenet_dataset,
            'mnist': get_mnist_dataset
        }[dataset.lower()]
    except KeyError:
        raise ValueError('Dataset {} not supported'.format(dataset))


def get_data_loaders(dataset_fn, data_dir, arch, batch_size, workers, validation_split,
                     effective_train_size, effective_valid_size, effective_test_size, test_only):
    """Create data loaders of selected size and random sampling
    """
    def split_list(list_to_split, ratio, shuffle=False):
        if shuffle:
            np.random.shuffle(list_to_split)
        split_idx = int(np.floor(ratio * len(list_to_split)))
        return list_to_split[:split_idx], list_to_split[split_idx:]

    # load the datasets
    train_dataset, test_dataset = dataset_fn(data_dir, arch, load_train=not test_only, load_test=True)

    # custom way of not using multiple processes
    if workers == 0:
        torch.set_num_threads(1)

    test_indices = list(range(len(test_dataset)))
    #test_sampler = SwitchingSubsetRandomSampler(test_indices, effective_test_size)
    test_sampler = torch.utils.data.RandomSampler(test_indices,
                                                  num_samples=int(effective_test_size * len(test_indices)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=workers,
        #   worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )
    if test_only:
        return None, None, test_loader

    if train_dataset is None:
        return None, test_loader, test_loader

    train_indices, valid_indices = split_list(list(range(len(train_dataset))), 1 - validation_split, shuffle=True)
    train_indices, valid_indices = list(train_indices), list(valid_indices)

    #train_sampler = SwitchingSubsetRandomSampler(train_indices, effective_train_size)
    train_sampler = torch.utils.data.RandomSampler(train_indices,
                                                   num_samples=int(effective_train_size * len(train_indices)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        # worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = None
    if valid_indices:
        #valid_sampler = SwitchingSubsetRandomSampler(valid_indices, effective_valid_size)
        valid_sampler = torch.utils.data.RandomSampler(valid_indices,
                                                       num_samples=int(effective_valid_size * len(valid_indices)))
        valid_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=workers,
            # worker_init_fn=worker_init_fn,
            pin_memory=True, 
            drop_last=True
        )

    return train_loader, valid_loader or test_loader, test_loader


### Dataset-specific functions 

def get_cifar10_dataset(cifar10_path, arch, load_train, load_test):
    """Load the CIFAR10 dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(
            root=cifar10_path, train=True, download=True, transform=train_transform
        )

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(
            root=cifar10_path, train=False, download=True, transform=test_transform
        )

    return train_dataset, test_dataset


def get_cifar100_dataset(cifar100_path, arch, load_train, load_test):
    """Load the CIFAR100 dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100(
            root=cifar100_path, train=True, download=True, transform=train_transform
        )

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR100(
            root=cifar100_path, train=False, download=True, transform=test_transform
        )

    return train_dataset, test_dataset

def get_imagenet_dataset(data_dir, arch, load_train=True, load_test=True):
    if 'inception' in arch.lower():
        resize, crop = 336, 299
    else:
        resize, crop = 256, 224
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ImagenetDataset(f"{data_dir}/ILSVRC2012_validation_label.txt",
                              f"{data_dir}/val", transform=transform)
    return None, test_dataset

# def get_imagenet_dataset(data_dir, arch, load_train=True, load_test=True):
#     """Load the ImageNet dataset according to:
#        https://github.com/pytorch/examples/blob/main/imagenet/main.py
#     """
#     if 'inception' in arch.lower():
#         resize, crop = 336, 299
#     else:
#         resize, crop = 256, 224
#
#     if 'googlenet' in arch.lower():
#         mean=[0.5, 0.5, 0.5]
#         std=[0.5, 0.5, 0.5]
#     else:
#         mean=[0.485, 0.456, 0.406]
#         std=[0.229, 0.224, 0.225]
#
#     train_dir = os.path.join(data_dir, 'train')
#     test_dir = os.path.join(data_dir, 'val')
#
#     train_dataset = None
#     if load_train:
#         train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(crop),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#              transforms.Normalize(mean, std),
#         ])
#         train_dataset = datasets.ImageFolder(train_dir, train_transform)
#
#     test_dataset = None
#     if load_test:
#         test_transform = transforms.Compose([
#             transforms.Resize(resize),
#             transforms.CenterCrop(crop),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])
#         test_dataset = datasets.ImageFolder(test_dir, test_transform)
#
#     return train_dataset, test_dataset


def get_tinyimagenet_dataset(data_dir, load_train=True, load_test=True):
    """Load tiny-imagenet 200 dataset"""
    def restructure_tiny_imagenet():
        val_dir = os.path.join(data_dir, 'val')

        # read annotations and assign images to their labels
        val_dict = {}
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                image_file, label, *_ = line.split('\t')
                val_dict[image_file] = label

        # put images in their respective label folders (create if they dont exist)
        for img, folder in val_dict.items():
            newpath = os.path.join(val_dir, folder)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_dir, 'images', img)):
                shutil.move(os.path.join(val_dir, 'images', img), os.path.join(newpath, img))

        # remove empty directory of original images
        os.rmdir(os.path.join(val_dir, 'images'))

    if os.path.isdir(os.path.join(data_dir, 'val', 'images')):
        restructure_tiny_imagenet()

    # Define transformation sequence for image pre-processing
    # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
    # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
    # std=[0.229, 0.224, 0.225])

    #transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    preprocess_transform_pretrain = transforms.Compose([
                    transforms.Resize(256), # Resize images to 256 x 256
                    transforms.CenterCrop(224), # Center crop image
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # Converting cropped images to tensors
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    #val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'val')

    train_dataset = None
    if load_train:
        train_dataset = datasets.ImageFolder(train_dir, preprocess_transform_pretrain)

    test_dataset = None
    if load_test:
        test_dataset = datasets.ImageFolder(test_dir, preprocess_transform_pretrain)

    return train_dataset, test_dataset


def get_mnist_dataset(data_dir, arch, load_train=True, load_test=True):
    """Load the MNIST dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True,
                                       download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      download=True, transform=test_transform)

    return train_dataset, test_dataset