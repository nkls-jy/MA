"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.

"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import rasterio

# keep
def sar_filter(filename):
    ext = os.path.splitext(filename)[1]
    return (ext == '.mat') or (ext == '.hdf5')

#keep
def image_filter(filename):
    from torchvision.datasets.folder import IMG_EXTENSIONS
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

# not used right now
def sar_loader(path):
    import h5py
    with h5py.File(path, 'r') as fid:
        data = np.copy(fid['data'][()])
    assert (data.shape[0] == 3)
    return data

#only for single band images
#def pil_loader(path):
#    with open(path, 'rb') as f:
#        img = np.asarray(Image.open(f))
#        #img = np.array(img.load())
#        print(img.shape)
#    return img

# image loader for 2-band tiffs
def img_loader(path):
    with rasterio.open(path) as f:
        img = f.read()
    #img = rasterio.open(path)
    #arr = img.read()
    #return arr
    return img

# keep
class NumpyToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x.copy(order='C'))

# keep
class CenterCropNy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        y1 = (img.shape[1] - self.size) // 2
        x1 = (img.shape[2] - self.size) // 2
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img[:, y1:y2, x1:x2]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

# keep
class RandomCropNy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        #print(f'img shape: {img.shape}')

        # create random origina crop values
        y1 = np.random.randint(0, img.shape[1] - self.size)
        x1 = np.random.randint(0, img.shape[2] - self.size)
      
        y2 = y1 + self.size
        x2 = x1 + self.size
        
        # return cropped images
        return img[:, y1:y2, x1:x2]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

# keep 
class Random8OrientationNy(object):
    def __init__(self):
        pass

    def __call__(self, img):
        k = img.shape[0]
        rot = np.random.randint(0, 8)

        # rotate along y and x axis
        img = np.rot90(img, axes=(1, 2), k=rot)
        if rot > 3:
            img = img[:, ::-1]
        assert (img.shape[0] == k)
        return img

# keep
def find_files(dir, filter=None):
    images = list()
    if filter is None:
        filter = lambda x: True

    #print(dir)
    for fname in sorted(os.listdir(dir)):
        if filter(fname):
            images.append(os.path.join(dir, fname))

    return images

# keep
class PlainImageFolder(Dataset):
    r"""
    Adapted from torchvision.datasets.folder.ImageFolder
    """

    def __init__(self, dirs, transform=None, cache=False, loader=img_loader, filter=image_filter):
        self.cache = cache
        self.img_cache = {}
        if isinstance(dirs, list):
            imgs = list()
            for r in dirs:
                imgs.extend(find_files(r, filter=filter))
        else:
            imgs = find_files(dirs, filter=filter)
            print(imgs)

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + dirs))

        self.dirs = dirs
        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        path = self.imgs[index]
        if not index in self.img_cache:
            img = self.loader(path)
            if self.cache:
                self.img_cache[index] = img
        else:
            img = self.img_cache[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

# keep
class PlainSarFolder(PlainImageFolder):
    def __init__(self, dirs, transform=None, cache=False):
        # testing
        PlainImageFolder.__init__(self, dirs, transform=transform, cache=cache, loader=img_loader, filter=image_filter)
        # original
        #PlainImageFolder.__init__(self, dirs, transform=transform, cache=cache, loader=sar_loader, filter=sar_filter)


# check if necessary
# if __name__ == "__main__":
#     from torchvision.transforms.functional import to_tensor
#     from PIL import Image

#     file = "/nas184/experiments/davide.cozzolino/otherMethos/visinf/n3net/datasets/sets/train400/16004.jpg"
#     img = Image.open(file).convert('L', (0.2989, 0.5870, 0.1140, 0))
#     t = to_tensor(img)


