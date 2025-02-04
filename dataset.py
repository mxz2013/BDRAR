import os
import os.path

import torch.utils.data as data
from PIL import Image
from misc import check_img_ext



def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'ShadowImages')) if f.endswith('.jpg') or f.endswith('jpg') or f.endswith('.jpeg')]
    img_shadow_pair = []
    for img_name in img_list:
        ext_img = check_img_ext(os.path.join(root, 'ShadowImages'), img_name)
        ext_msk = check_img_ext(os.path.join(root, 'ShadowMasks'), img_name)
        img_shadow_pair.append((os.path.join(root, 'ShadowImages', img_name + ext_img), os.path.join(root, 'ShadowMasks', img_name + ext_msk)))

    return img_shadow_pair


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
