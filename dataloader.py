def create_train_realsar_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=folders_data.train_mlook_dir, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=20)

    return trainloader

def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=folders_data.valid_mlook_dir, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=1)

    return validloader