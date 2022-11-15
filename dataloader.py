import torch
from dataset import sar_dataset
from torchvision.transforms import Compose

# maybe not needed?
#scale_img = 255.0

# paths
dir_path = "T:\\Jaggy\\Masterarbeit\\DataPreparation\\valid\\"

def create_train_realsar_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=dir_path, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader

def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=dir_path, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader

class PreprocessingIntNoisyFromAmp:
    def __init__(self, flag_bayes=False):
        from torch.distributions.gamma import Gamma
        self.gen_dist = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        self.flag_bayes = flag_bayes
    def __call__(self, target):
        if self.flag_bayes:
            target = target + 1 / scale_img
        target = target ** 2
        noise = self.gen_dist.sample(target.shape)[:, :, :, :, 0]
        mask  = torch.ones(target.shape)
        if target.is_cuda:
            noise = noise.cuda()
            mask  = mask.cuda()
        noisy = target * noise
        return noisy, target, mask

'''
if __name__ == '__main__':
    #data_iterator = create_valid_realsar_dataloaders(256, 8)
    data_iterator = create_train_realsar_dataloaders(256, 32, 1)

    import matplotlib.pyplot as plt
    for index, patch in enumerate(data_iterator):
        noisy, target, mask = data_preprocessing(patch)
        if flag_log:
            noisy = noisy.exp()
            target = target.exp()
        else:
            noisy = noisy.sqrt()
            target = target.sqrt()
        print(index, patch.shape, noisy.shape)
        plt.figure()
        plt.subplot(1,3,1); plt.imshow(noisy[0,0] , clim=[0, 1], cmap='gray')
        plt.subplot(1,3,2); plt.imshow(target[0,0], clim=[0, 1],cmap='gray')
        plt.subplot(1,3,3); plt.imshow(mask[0,0]  , clim=[0, 1], cmap='gray')
        plt.show()
'''