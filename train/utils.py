import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim




def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, star_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = star_epoch
        self.epoch_re = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch_re-1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():

                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')

                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.epoch_re += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(SSIM_Loss, self).forward(img1, img2))
def delta_e_loss(y_pred, y_true,n=2):
    # 计算△L*
    L_pred = 0.299 * y_pred[:, 0, :, :] + 0.587 * y_pred[:, 1, :, :] + 0.114 * y_pred[:, 2, :, :]
    L_true = 0.299 * y_true[:, 0, :, :] + 0.587 * y_true[:, 1, :, :] + 0.114 * y_true[:, 2, :, :]
    delta_L = L_pred - L_true

    # 计算△a*和△b*
    a_pred = -0.147 * y_pred[:, 0, :, :] - 0.289 * y_pred[:, 1, :, :] + 0.436 * y_pred[:, 2, :, :]
    a_true = -0.147 * y_true[:, 0, :, :] - 0.289 * y_true[:, 1, :, :] + 0.436 * y_true[:, 2, :, :]
    delta_a = a_pred - a_true

    b_pred = 0.615 * y_pred[:, 0, :, :] - 0.515 * y_pred[:, 1, :, :] + 0.100 * y_pred[:, 2, :, :]
    b_true = 0.615 * y_true[:, 0, :, :] - 0.515 * y_true[:, 1, :, :] + 0.100 * y_true[:, 2, :, :]
    delta_b = b_pred - b_true

    # 计算△E*
    delta_E = torch.sqrt(delta_L ** 2 + delta_a ** 2 + delta_b ** 2+1e-8)

    # 对 batch 维求平均，作为最终的 loss
    return delta_E.mean()*n
def delta_e_loss_xyz(y_pred, y_true):
    # 计算△L*
    X_pred = 0.49 * y_pred[:, 0, :, :] + 0.31 * y_pred[:, 1, :, :] + 0.20 * y_pred[:, 2, :, :]
    X_true = 0.49 * y_true[:, 0, :, :] + 0.31 * y_true[:, 1, :, :] +0.20 * y_true[:, 2, :, :]
    delta_X = (1/0.17697)*(X_pred - X_true)

    # 计算△a*和△b*
    Y_pred = 0.17697 * y_pred[:, 0, :, :] + 0.81240 * y_pred[:, 1, :, :] + 0.01063 * y_pred[:, 2, :, :]
    Y_true = 0.17697 * y_true[:, 0, :, :] + 0.81240 * y_true[:, 1, :, :] + 0.01063 * y_true[:, 2, :, :]
    delta_Y = (1/0.17697)*(Y_pred - Y_true)

    Z_pred = 0.00 * y_pred[:, 0, :, :] + 0.01 * y_pred[:, 1, :, :] + 0.99 * y_pred[:, 2, :, :]
    Z_true = 0.00 * y_true[:, 0, :, :] + 0.01 * y_true[:, 1, :, :] + 0.99 * y_true[:, 2, :, :]
    delta_Z = (1/0.17697)*(Z_pred - Z_true)

    # 计算△E*
    delta_E = torch.sqrt(delta_X ** 2 + delta_Y ** 2 + delta_Z ** 2+1e-8)

    # 对 batch 维求平均，作为最终的 loss
    return 0.4*delta_E.mean()
class ReplayBuffer_D():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
        self.data1 = []

    def push_and_pop(self, data,data1):
        to_return = []
        to_return1 = []
        n=0
        a=data.data
        for element in data.data:
            j = 0
            n=n+1
            for element1 in data1.data:
                j=j+1
                if n==j:
                    element = torch.unsqueeze(element, 0)
                    element1 = torch.unsqueeze(element1, 0)
                    if len(self.data) < self.max_size:
                        self.data.append(element)
                        to_return.append(element)
                        self.data1.append(element1)
                        to_return1.append(element1)
                    else:
                        if random.uniform(0, 1) > 0.5:
                            i = random.randint(0, self.max_size-1)
                            to_return.append(self.data[i].clone())
                            to_return1.append(self.data1[i].clone())
                            self.data[i] = element
                            self.data1[i] = element1
                        else:
                            to_return.append(element)
                            to_return1.append(element1)
        return Variable(torch.cat(to_return)),Variable(torch.cat(to_return1))
