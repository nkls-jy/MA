"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.

"""

import torch
import os
from utils import metrics
from tqdm import tqdm
import time
import rasterio
import numpy as np

def save_checkpoint(experiment):
    # Save checkpoint

    net = experiment.net
    optimizer = experiment.optimizer
    expdir = experiment.expdir
    epoch = experiment.epoch

    state = {
        'net': net.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch
    }
    checkpoint_dir = os.path.join(expdir, "checkpoint/")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = '%03d_ckpt.t7' % (epoch)
    print('Saving ', filename)
    torch.save(state, os.path.join(checkpoint_dir, filename))


def load_checkpoint(experiment, epoch=-1, withoptimizer=False):
    # load checkpoint
    checkpoint_dir = os.path.join(experiment.expdir, "checkpoint/")

    if epoch < 0:
        for i in range(1310):
            filename = '%03d_ckpt.t7' % i
            if not os.path.exists(os.path.join(checkpoint_dir, filename)):
                break
        epoch = i - 1

    filename = os.path.join(checkpoint_dir, '%03d_ckpt.t7' % epoch)
    print("Loading ", filename)
    checkpoint = torch.load(filename)
    experiment.net.load_state_dict(checkpoint["net"])
    if withoptimizer:
        experiment.optimizer.load_state_dict(checkpoint["optim"])

    return epoch


def set_random_seeds(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

# 1 training epoch
def train_epoch(experiment, trainloader, data_preprocessing, log_data):
    lr = experiment.base_lr * experiment.learning_rate_decay(experiment.epoch)
    if lr == 0:
        return True
    for group in experiment.optimizer.param_groups:
        group['lr'] = lr
    print('\nEpoch: %d, Learning rate: %f, Expdir %s' % (experiment.epoch, lr, experiment.expname))

    # sets model into training mode
    experiment.net.train()

    stats_num = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}
    stats_cum = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}

    for inputs in tqdm(trainloader):
        if experiment.use_cuda:
            # transfers inputs from tensor to GPU
            inputs = inputs.cuda()
        
        # sets all gradients of all tensors to zero
        experiment.optimizer.zero_grad()

        # 
        if log_data:
            noisy_log, target_log = data_preprocessing(inputs)
            noisy = experiment.preprocessing_log2net(noisy_log)
            target = experiment.preprocessing_log2net(target_log)
            target_amp = target_log.exp()
            del noisy_log
            del target_log
        else:
            noisy_int, target_int = data_preprocessing(inputs)
            noisy = experiment.preprocessing_int2net(noisy_int)
            target = experiment.preprocessing_int2net(target_int)
            target_amp = target_int.abs().sqrt()
            del noisy_int
            del target_int

        # model prediction
        pred = experiment.net(noisy)

        #print(f"target shape[1]: {target.shape[1]}")
        #print(f"target shape[2]: {target.shape[2]}")
        #print(f"target shape[3]: {target.shape[3]}")

        # padding handling?
        # old:        
        #pad_row = (target.shape[1] - pred.shape[1]) // 2
        #pad_col = (target.shape[2] - pred.shape[2]) // 2

        # new:
        pad_row = (target.shape[2] - pred.shape[2]) // 2
        pad_col = (target.shape[3] - pred.shape[3]) // 2
        #print(f"pad_row new: {pad_row}")
        #print(f"pad_col new: {pad_col}")

        # account for even or uneven row/column sizes?
        if pad_row > 0:
            # old:
            #target = target[:, pad_row: -pad_row, :]
            #target_amp = target_amp[:, pad_row: -pad_row, :]
            # new:
            target = target[:, :, pad_row: -pad_row, :]
            target_amp = target_amp[:, :, pad_row: -pad_row, :]
        if pad_col > 0:
            # old:
            #target = target[:, :, pad_col: -pad_col]  # .contiguous()
            #target_amp = target_amp[:, :, pad_col: -pad_col]  # .contiguous()
            # new:
            target = target[:, :, :, pad_col: -pad_col]
            target_amp = target_amp[:, :, :, pad_col: -pad_col]
        
        #print(f"target size: {target.size()}")
        #print(f"pred size: {pred.size()}")

        #print(f"gradient of layer 0: {experiment.net[0].weight.grad}")
        
        # calculate loss
        loss = experiment.criterion(pred, target).mean()

        # creates a loop where every tensor has requires_grad = False
        # any tensor with gradient currently attached with current computational graph is not detached from graph
        with torch.no_grad():
            pred_amp = experiment.postprocessing_net2amp(pred.detach())

            stats_one = dict()
            stats_one["loss"] = loss.data
            stats_one["psnr"] = metrics.metric_psnr(pred_amp, target_amp, maxval=1.0, size_average=True).data
            stats_one["mse"] = metrics.metric_mse(pred_amp, target_amp, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim(pred_amp, target_amp, size_average=True).data
        

        
        # backpropagate loss
        loss.backward()
        del loss
        del pred
        del pred_amp

        # update weights
        experiment.optimizer.step()
        
        # update stats
        for stats_key in stats_cum:
            stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
            stats_num[stats_key] = stats_num[stats_key] + 1

    for stats_key in stats_cum:
        stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]

    print('Epoch: %d |' % experiment.epoch, end='')
    experiment.add_summary("train/epoch", experiment.epoch)
    for stats_key, stats_value in stats_cum.items():
        print(' %5s: %.5f | ' % (stats_key, stats_value), end='')
        experiment.add_summary("train/" + stats_key, stats_value)
    print("")

    experiment.epoch += 1
    return False


def test_epoch(experiment, testloader, data_preprocessing, log_data):
    stats_num = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}
    stats_cum = {"loss": 0, "mse": 0, "psnr": 0, "ssim": 0}

    experiment.net.eval()
    with torch.no_grad():

        print("Testing ")
        for inputs in tqdm(testloader):
            torch.cuda.empty_cache()

            if experiment.use_cuda:
                inputs = inputs.cuda()

            if log_data:
                noisy_log, target_log, mask = data_preprocessing(inputs)
                noisy = experiment.preprocessing_log2net(noisy_log)
                target = experiment.preprocessing_log2net(target_log)
                target_amp = target_log.exp()
            else:
                noisy_int, target_int, mask = data_preprocessing(inputs)
                noisy = experiment.preprocessing_int2net(noisy_int)
                target = experiment.preprocessing_int2net(target_int)
                target_amp = target_int.abs().sqrt()

            noisy = torch.autograd.Variable(noisy, requires_grad=False)
            target = torch.autograd.Variable(target, requires_grad=False)
            
            # prediction
            pred = experiment.net(noisy)

            pad_row = (target.shape[2] - pred.shape[2]) // 2
            pad_col = (target.shape[3] - pred.shape[3]) // 2
            if pad_row > 0:
                target = target[:, :, pad_row:-pad_row, :]
                target_amp = target_amp[:, :, pad_row:-pad_row, :]
            if pad_col > 0:
                target = target[:, :, :, pad_col: -pad_col].contiguous()
                target_amp = target_amp[:, :, :, pad_col: -pad_col].contiguous()

            batch_loss = experiment.criterion(pred, target)

            loss = batch_loss.mean()

            pred_amp = experiment.postprocessing_net2amp(pred)

            stats_one = dict()
            stats_one["loss"] = loss.data
            stats_one["psnr"] = metrics.metric_psnr_mask(pred_amp, target_amp, mask, maxval=1.0, size_average=True).data
            stats_one["mse"] = metrics.metric_mse_mask(pred_amp, target_amp, mask, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim_mask(pred_amp, target_amp, mask, size_average=True).data

            for stats_key in stats_cum:
                stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                stats_num[stats_key] = stats_num[stats_key] + 1

            del pred, noisy, target

        for stats_key in stats_cum:
            stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]

        experiment.add_summary("test/epoch", experiment.epoch)
        for stats_key, stats_value in stats_cum.items():
            print(' %5s: %.5f | ' % (stats_key, stats_value), end='')
            experiment.add_summary("test/" + stats_key, stats_value)
        print("")

    return stats_cum


def trainloop(experiment, trainloader, data_preprocessing, log_data, validloader=None):
    stop = False
    while not stop:
        save_checkpoint(experiment)
        if validloader is not None:
            test_epoch(experiment, validloader, data_preprocessing, log_data)
        stop = train_epoch(experiment, trainloader, data_preprocessing, log_data)


def test_list(experiment, outdir, listfile, pad=0):
    net = experiment.net
    # old:
    #eval_file = os.path.join(outdir, "result_%s.mat")
    # new:
    eval_file = os.path.join(outdir, "result_%s")
    os.makedirs(outdir, exist_ok=True)
    use_cuda = experiment.use_cuda

    net.eval()
    
    #stats_num = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    #stats_cum = {"mse": 0, "psnr": 0, "ssim": 0}
    vetTIME = list()
    print(f"tesetfiles: {listfile}")
    
    with torch.no_grad():
        for filename in listfile:
            with rasterio.open(filename) as f:
                img = f.read()
                # args for output
                kwargs = f.meta

                outending = filename.rsplit('/', 1)[1]

            output_filename = eval_file & outending

            print(f"img shape: {img.shape}")

            noisy_int = img

            timestamp = time.time()

            noisy_int = torch.from_numpy(noisy_int)[None, :, :]

            if use_cuda:
                noisy_int = noisy_int.cuda()
            if pad>0:
                noisy_int = torch.nn.functional.pad(noisy_int, (pad, pad, pad, pad), mode='reflect', value=0)
            
            noisy = experiment.preprocessing_int2net(noisy_int)
            pred = net(noisy)

            pred_int = experiment.postprocessing_net2int(pred)[0, 0, :, :]
            if use_cuda:
                pred_int = pred_int.cpu()
            vetTIME.append(time.time()-timestamp)

            # create two band output array 
            pred_img = pred_int.numpy()[np.newaxis, :, :]
            img = np.squeeze(img)[np.newaxis, :, :]
            print(f"pred shape: {pred_img.shape}")
            print(f"orig. shape: {img.shape}")

            #pad_row = (img.shape[0] - pred_img.shape[0]) // 2
            pad_row = (pred_img.shape[1] - img.shape[1]) // 2
            #pad_col = (img.shape[1] - pred_img.shape[1]) // 2
            pad_col = (pred_img.shape[2] - img.shape[2]) // 2

            print(f"pad row: {pad_row}")
            print(f"pad_col: {pad_col}")
            if pad_row > 0:
                pred_img = pred_img[:, pad_row: -pad_row, :]
            if pad_col > 0:
                pred_img = pred_img[:, :, pad_col: -pad_col]      
            
            print(f"row & col values now: {pred_img.shape}")
        
            outfile = np.append(pred_img, img, axis=0)
            print(f"outfile shape: {outfile.shape}")

            # write output file (TIFF)
            kwargs.update(
                dtype=rasterio.float32,
                count=2,
                compress='lzw')

            with rasterio.open(output_filename, 'w', **kwargs) as dst:
                dst.write(outfile.astype(rasterio.float32))


            '''
            target_int = torch.from_numpy(dat['target_int'])[None, None, :, :]
            target_amp = target_int.abs().sqrt()
            mask = torch.from_numpy(dat['mask'])[None, None, :, :]
            if use_cuda:
                mask = mask.cuda()
                target_amp = target_amp.cuda()
            pred_amp = experiment.postprocessing_net2amp(pred)

            pad_row = (target_amp.shape[2] - pred_amp.shape[2]) // 2
            pad_col = (target_amp.shape[3] - pred_amp.shape[3]) // 2
            if pad_row > 0:
                target_amp = target_amp[:, :, pad_row: -pad_row, :]
                mask = mask[:, :, pad_row: -pad_row, :]
            if pad_col > 0:
                target_amp = target_amp[:, :, :, pad_col: -pad_col]
                mask = mask[:, :, :, pad_col: -pad_col].contiguous()
            if pad_col != 0 or pad_row != 0:
                print('error_size', pad_col, pad_row)

            stats_one = dict()
            stats_one["mse"] = metrics.metric_mse_mask(pred_amp, target_amp, mask, size_average=True).data
            stats_one["psnr"] = metrics.metric_psnr_mask(pred_amp, target_amp, mask, maxval=1.0, size_average=True).data
            stats_one["ssim"] = metrics.metric_ssim_mask(pred_amp, target_amp, mask, size_average=True).data

            for stats_key, stats_value in stats_one.items():
                print(' %5s: %.5f | ' % (stats_key, stats_value), end='')
            print("")
            for stats_key in stats_cum:
                stats_cum[stats_key] = stats_cum[stats_key] + stats_one[stats_key]
                stats_num[stats_key] = stats_num[stats_key] + 1

    for stats_key in stats_cum:
        stats_cum[stats_key] = stats_cum[stats_key] / stats_num[stats_key]
    print(' %7s |' % 'AVG', end='')
    for stats_key, stats_value in stats_cum.items():
        print(' %5s: %.5f | ' % (stats_key, stats_value), end='')
    print("")
    savemat(os.path.join(outdir, "info.mat"), {'vetTIME': vetTIME})
    '''

def test_list_weights(experiment, outdir, listfile, pad=0):
    net = experiment.net
    eval_file = os.path.join(outdir, "weights_%s.mat")
    os.makedirs(outdir, exist_ok=True)
    use_cuda = experiment.use_cuda

    net.eval()
    from scipy.io import loadmat, savemat
    with torch.no_grad():
        for name, filename in listfile:
            dat = loadmat(filename)
            output_filename = eval_file % name

            print(' %7s |' % name, output_filename, end='')
            noisy_int = dat['noisy_int']
            timestamp = time.time()
            noisy_int = torch.from_numpy(noisy_int)[None, None, :, :]
            if use_cuda:
                noisy_int = noisy_int.cuda()

            if pad > 0:
                noisy_int = torch.nn.functional.pad(noisy_int, (pad, pad, pad, pad), mode='reflect', value=0)

            noisy = experiment.preprocessing_int2net(noisy_int)
            weights = net.forward_weights(noisy, reshape=True)

            datout = dict()
            datout['weights'] = weights.cpu().numpy()
            savemat(output_filename, datout)

            print(" done")


if __name__ == '__main__':
    from scipy.io import loadmat, savemat
    import numpy as np

    norm_int = 65536.0
    for index in range(1, 13):
        # filename = '/datasets/deepSAR/code_CNN/code_matlab_sync/data/Test/Set12/%02d.png.mat' % index
        filename = '/datasets/deepSAR/code_IGARSS2017/matlab_code/other_syn/Set12/%02d.png.mat' % index
        output_filename = '../datatest/synt%02d.mat' % index

        dat = loadmat(filename)
        print(dat.keys())
        print(dat['label'].dtype, dat['label'].min(), dat['label'].max())
        print(dat['input'].dtype, dat['input'].min(), dat['input'].max())
        target_int = np.square(dat['label']).astype(np.float32) / norm_int
        noisy_int = np.square(dat['input']).astype(np.float32) / norm_int

        datout = dict()
        datout['target_int'] = target_int
        datout['noisy_int'] = noisy_int
        datout['norm_int'] = norm_int
        datout['mask'] = np.ones_like(noisy_int)
        # savemat(output_filename, datout)

