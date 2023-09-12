import argparse
import itertools
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from models import ResNet_seg,Discriminator
from utils import ReplayBuffer_D
from utils import LambdaLR
from utils import Logger, ReplayBuffer
from utils import weights_init_normal
from utils import MS_SSIM_Loss,delta_e_loss_xyz
from datasets import ImageDataset_label, ImageDataset

# breast/ neuroendocrine / GLAS
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../datasets/big_image/train', help='root directory of the dataset')
    parser.add_argument('--modelroot', type=str, default='../checkpoint/dsff', help='root directory of the model')
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=11, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=2, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    parser.add_argument('--continue_train', type=bool, default=False, help='load model and continue trainning')
    parser.add_argument('--loadroot', type=str, default='', help='continue train root directory of the model')
    opt = parser.parse_args()
    print(opt)


    # Fix random seed for reproducibility
    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    same_seeds(100)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = ResNet_seg()
    netG_B2A = ResNet_seg()
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    # Lossess
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = nn.DataParallel(netG_A2B, device_ids=[0,1, 2, 3]).cuda()
        netG_B2A = nn.DataParallel(netG_B2A, device_ids=[0,1, 2, 3]).cuda()
        netD_A = nn.DataParallel(netD_A, device_ids=[ 0,1,2, 3]).cuda()
        netD_B = nn.DataParallel(netD_B, device_ids=[ 0,1,2, 3]).cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
        criterion_ssim.cuda()

    netG_A2B.train()
    netG_B2A.train()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)


    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    expert_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    expert_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    label = Tensor(opt.batchSize, 1, opt.size // 8, opt.size // 8)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()


    dataloader = DataLoader(ImageDataset(opt.dataroot, batch_size=opt.batchSize ),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    expertloader = DataLoader(ImageDataset_label(opt.dataroot, batch_size=opt.batchSize),
                              batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)



    ###################################
    start_epoch = opt.epoch
    pre_trainning = 1
    if opt.continue_train:
        pre_trainning = 0
        netG_A2B_checkpoint = torch.load(os.path.join(opt.loadroot, 'netG_A2B.pth'))  # 加载断点
        netG_A2B.load_state_dict(netG_A2B_checkpoint['model'])  # 加载模型可学习参数

        netG_B2A_checkpoint = torch.load(os.path.join(opt.loadroot, 'netG_B2A.pth'))  # 加载断点
        netG_B2A.load_state_dict(netG_B2A_checkpoint['model'])  # 加载模型可学习参数
        optimizer_G.load_state_dict(netG_B2A_checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler_G.load_state_dict(netG_B2A_checkpoint['lr_schedule'])  # 加载lr_scheduler
        start_epoch = netG_B2A_checkpoint['epoch']

        netD_A_checkpoint = torch.load(os.path.join(opt.loadroot, 'netD_A.pth'))  # 加载断点
        netD_A.load_state_dict(netD_A_checkpoint['model'])  # 加载模型可学习参数
        optimizer_D_A.load_state_dict(netD_A_checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler_D_A.load_state_dict(netD_A_checkpoint['lr_schedule'])  # 加载lr_scheduler

        netD_B_checkpoint = torch.load(os.path.join(opt.loadroot, 'netD_B.pth'))  # 加载断点
        netD_B.load_state_dict(netD_B_checkpoint['model'])  # 加载模型可学习参数
        optimizer_D_B.load_state_dict(netD_B_checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler_D_B.load_state_dict(netD_B_checkpoint['lr_schedule'])  # 加载lr_scheduler

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader), start_epoch)
    expert_iter = iter(expertloader)
    ###### Training ######
    for epoch in range(start_epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['HE']))
            real_B = Variable(input_B.copy_(batch['Ki67']))

            try:
                expert_batch = expert_iter.__next__()
            except StopIteration:
                expert_iter = iter(expertloader)
                expert_batch = expert_iter.__next__()

            real_expert_A = Variable(expert_A.copy_(expert_batch['HE']))
            real_expert_B = Variable(expert_B.copy_(expert_batch['Ki67']))

            real_expert = Variable(label.copy_(expert_batch['Label']))

            # Generators A2B and B2A
            optimizer_G.zero_grad()

            c_expa = netG_A2B(real_expert_A, mode='C')
            loss_expert_A = criterion_GAN(c_expa, real_expert)

            c_expb = netG_B2A(real_expert_B, mode='C')
            loss_expert_B = criterion_GAN(c_expb, real_expert)

            loss_C = (loss_expert_A + loss_expert_B)

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B, _, _= netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)
            # G_B2A(A) should equal A if real A is fed
            same_A, _, _ = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)

            # GAN loss
            fake_B, features_lfb, c_fb = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A, features_lfa, c_fa = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A, features_lra, c_ra = netG_B2A(fake_B)
            loss_cycle_embd_ABA = criterion_cycle(features_lra, features_lfb)

            loss_class_pathology_ABA = criterion_GAN(c_ra, c_fb)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
            loss_cycle_ssimABA = criterion_ssim(recovered_A, real_A)
            loss_cycle_colorABA = delta_e_loss_xyz(recovered_A, real_A)


            recovered_B, features_lrb, c_rb = netG_A2B(fake_A)
            loss_cycle_embd_BAB = criterion_cycle(features_lrb, features_lfa)

            loss_class_pathology_BAB = criterion_GAN(c_rb, c_fa)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)
            loss_cycle_ssimBAB = criterion_ssim(recovered_B, real_B)
            loss_cycle_colorBAB = delta_e_loss_xyz(recovered_B, real_B)

            # Total loss
            loss_G = 5.0 * loss_C +\
                     5.0 * (loss_identity_A + loss_identity_B) + \
                     2.0 * (loss_GAN_A2B + loss_GAN_B2A) + \
                     10.0 * (loss_cycle_ABA + loss_cycle_BAB) + \
                     2.0 * (loss_cycle_ssimABA + loss_cycle_ssimBAB) + \
                     2.0 * (loss_cycle_colorABA + loss_cycle_colorBAB) + \
                     1.0 * (loss_cycle_embd_ABA + loss_cycle_embd_BAB) + \
                     1.0 * (loss_class_pathology_ABA + loss_class_pathology_BAB)

            loss_G.backward()

            optimizer_G.step()
            ###################################

            # Discriminator A
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss

            fake_Ad = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_Ad)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            # Discriminator B
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_Bd = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_Bd)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G,
                        'loss_G_idt': (loss_identity_A + loss_identity_B),
                        'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                        'loss_G_ssim': (loss_cycle_ssimABA + loss_cycle_ssimBAB),
                        'loss_G_color': (loss_cycle_colorABA + loss_cycle_colorBAB),
                        'loss_ps': (loss_class_pathology_ABA + loss_class_pathology_BAB),
                        'loss_embd': (loss_cycle_embd_ABA + loss_cycle_embd_BAB),
                        'loss_C': (loss_expert_A + loss_expert_B),
                        'loss_D': (loss_D_A + loss_D_B)},
                       images={'real_cycleGAN_pf_seg_A': real_A, 'real_cycleGAN_pf_seg_B': real_B,
                               'fake_cycleGAN_pf_seg_A': fake_A, 'fake_cycleGAN_pf_seg_B': fake_B,
                               'real_mask_A':label,
                               'G_sA': c_fb, 'G_sB': c_ra})

            # save models at half of an epoch
            if i == 200 and epoch == 1:
                saveroot = os.path.join(opt.modelroot, 'star_temp')
                if not os.path.exists(saveroot):
                    os.makedirs(saveroot)

                # Save models checkpoints
                netG_A2B_checkpoints = {
                    "model": netG_A2B.state_dict()
                }
                torch.save(netG_A2B_checkpoints, os.path.join(saveroot, 'netG_A2B.pth'))

                netG_B2A_checkpoints = {
                    "model": netG_B2A.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_scheduler_G.state_dict()
                }
                torch.save(netG_B2A_checkpoints, os.path.join(saveroot, 'netG_B2A.pth'))

                netD_A_checkpoints = {
                    "model": netD_A.state_dict(),
                    'optimizer': optimizer_D_A.state_dict(),
                    'lr_schedule': lr_scheduler_D_A.state_dict()
                }
                torch.save(netD_A_checkpoints, os.path.join(saveroot, 'netD_A.pth'))

                netD_B_checkpoints = {
                    "model": netD_B.state_dict(),
                    'optimizer': optimizer_D_B.state_dict(),
                    'lr_schedule': lr_scheduler_D_B.state_dict()
                }
                torch.save(netD_B_checkpoints, os.path.join(saveroot, 'netD_B.pth'))

            if (i+1) % (dataloader.__len__()//6 + 1) == 0:
                saveroot = os.path.join(opt.modelroot, 'temp')
                if not os.path.exists(saveroot):
                    os.makedirs(saveroot)

                # Save models checkpoints
                netG_A2B_checkpoints = {
                    "model": netG_A2B.state_dict()
                }
                torch.save(netG_A2B_checkpoints, os.path.join(saveroot, 'netG_A2B.pth'))

                netG_B2A_checkpoints = {
                    "model": netG_B2A.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_scheduler_G.state_dict()
                }
                torch.save(netG_B2A_checkpoints, os.path.join(saveroot, 'netG_B2A.pth'))

                netD_A_checkpoints = {
                    "model": netD_A.state_dict(),
                    'optimizer': optimizer_D_A.state_dict(),
                    'lr_schedule': lr_scheduler_D_A.state_dict()
                }
                torch.save(netD_A_checkpoints, os.path.join(saveroot, 'netD_A.pth'))

                netD_B_checkpoints = {
                    "model": netD_B.state_dict(),
                    'optimizer': optimizer_D_B.state_dict(),
                    'lr_schedule': lr_scheduler_D_B.state_dict()
                }
                torch.save(netD_B_checkpoints, os.path.join(saveroot, 'netD_B.pth'))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        saveroot = os.path.join(opt.modelroot, 'epoch'+str(epoch))
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), os.path.join(saveroot, 'netG_A2B.pth'))
        torch.save(netG_B2A.state_dict(), os.path.join(saveroot, 'netG_B2A.pth'))
        torch.save(netD_A.state_dict(), os.path.join(saveroot, 'netD_A.pth'))
        torch.save(netD_B.state_dict(), os.path.join(saveroot, 'netD_B.pth'))

