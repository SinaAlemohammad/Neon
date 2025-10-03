import warnings
warnings.filterwarnings("ignore")

import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build training set
    train_root = osp.join(data_path, 'train')
    train_set = DatasetFolder(
        root=train_root,
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_aug
    )
    
    # build validation set if present
    val_root = osp.join(data_path, 'val')
    try:
        val_set = DatasetFolder(
            root=val_root,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=val_aug
        )
    except FileNotFoundError:
        
        val_set = None

    num_classes = 1000

    # logging
    
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print('')
        
