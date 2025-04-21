import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark=True
import argparse
import time 
import torchvision
from datetime import datetime
os.environ["CUDA_VISIBIE_DEVICES"] ="0,1,2,3"
from loss.distortion import *
from net.network import SwinJSCC
# import torch.nn.DataParallel

parser = argparse.ArgumentParser(description='SwinJSCC')
parser.add_argument('--training', action='store_true', help='training or not')
parser.add_argument('--trainset', type=str, default='CIFAR10', choices=['CIFAR10'], help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak', choices=['kodak'], help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--model', type=str, default='SwinJSCC_w/_SAandRA', choices=['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'], help='SwinJSCC model or SwinJSCC without channel ModNet or rate ModNet')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='96', help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='10', help='random or fixed snr')
parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], help='SwinJSCC model size')
args = parser.parse_args()

class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 150

    test_data_dir = "/home/daicheng/mvcd_dataset/"

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/home/daicheng/datasets/"
        # test_data_dir = "/home/daicheng/datasets/"
    
        batch_size = 2
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )



CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()


def load_weights(model_path):
    pretrained = torch.load(model_path)
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(pretrained, strict=True)
    else:
        net.load_state_dict(pretrained, strict=True)
    del pretrained

def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step

    for batch_idx, (input, label) in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input = input.cuda()
        recon_image, CBR, SNR, mse, loss_G = net(input)

        loss = loss_G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)
        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)
        else:
            psnrs.update(100)
            msssims.update(100)

        if (global_step % config.print_step) == 0:
            # epoch = global_step // train_loader.__len()
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {cur_lr}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                if args.testset == 'kodak':
                    

                    for batch_idx, (input, label) in enumerate(test_loader):
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

                        save_dir = "./test_result"
                        for idx in range(recon_image.size(0)):  # 遍历 batch 中的每张图像
                            save_path = os.path.join(save_dir, f"recon_image_SNR{SNR}_rate{rate}_idx{batch_idx}_{idx}.png")
                            torchvision.utils.save_image(recon_image[idx].clamp(0., 1.), save_path)

                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                            psnrs.update(psnr.item())
                            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)
                        else:
                            psnrs.update(100)
                            msssims.update(100)

                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=False)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = SwinJSCC(args, config).cuda()

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for training.")
    #     net = torch.nn.DataParallel(net)
    print(f"Testing on Kodak dataset: {args.testset}")


    if not args.training:
        # model_path = "/home/daicheng/SwinJSCC-test/checkpoint/SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"
        model_path = '/home/daicheng/SwinJSCC_w_ACandBC_w_RAandSA/history/2025-04-07 21:06:33/models/2025-04-07 21:06:33_EP5.model'
        load_weights(model_path)

    # net = net.cuda()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    # if args.testset == 'kodak':
    #     test_loader = get_loader(config.test_data_dir, batch_size=config.batch_size)
    # elif args.testset == 'CIFAR10':
    #     test_loader = get_loader(config.test_data_dir, batch_size=config.batch_size)

    print(f"Test loader size:  {len(test_loader.dataset)}")


    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // test_loader.__len__()
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                test()
    else:
        test()

