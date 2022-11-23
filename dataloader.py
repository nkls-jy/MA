import torch
from dataset import sar_dataset
from torchvision.transforms import Compose
import torchvision

# maybe not needed?
#scale_img = 255.0

# paths
#train_path = ".\\sets\\train\\"
#valid_path = ".\\sets\\valid\\"

train_path = ".\\sets\\train_bands\\"


def create_train_realsar_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=train_path, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    print(f'trainset length: {len(trainset)}')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1) # 1 just for testing, usually 20)

    return trainloader

def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=valid_path, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader

class PreprocessingBatch:
    def __init__(self):
        from torch.distributions.gamma import Gamma
        self.gen_dist = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        self.flag_bayes = False # testing, usually taken from args, flag_bayes
    def __call__(self, batch):
        print(f'preprocessing input: {batch.shape}')
        
        noisy = images[:, 0, :, :]
        target = images[:, 1, :, :]
        
        if batch.is_cuda:
            noisy = noisy.cuda()
            target = target.cuda()
        
        # returns 2 tensors with all noisy/target images from batch
        return noisy, target

if __name__ == '__main__':
    #data_iterator = create_valid_realsar_dataloaders(256, 8)
    data_loader = create_train_realsar_dataloaders(25, 5, 1)
    data_preprocessing = PreprocessingBatch(); flag_log = False

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import numpy as np
    import torchvision.transforms.functional as F

    

    data_iter = iter(data_loader)
    images = data_iter.next()

    noisy, target = data_preprocessing(images)

    #noise = images[:, 0, :, :]
    print(noisy.shape)
    print(type(noisy))

    '''
    noise_list = images[:, 0, :, :].tolist()
    target_list = images[:, 1, :, :].tolist()

    print(len(noise_list))

    def show_images(images, cols = 1, titles = None):
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            #if image.ndim == 2:
            #   plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    show_images(noise_list)
    '''
    
