# By mama
# TIME 2023/3/22   17:25
import math
import functools
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.InstanceNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):
        _, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_c = torch.mean(x, dim=1, keepdim=True)
        x_c_max, _ = torch.max(x, dim=1, keepdim=True)
        x_c = torch.cat([x_c, x_c_max], dim=1)
        x_c = self.conv1(x_c)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)*self.sigmoid(x_c)
        return out
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out
class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, upsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 =nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False))
        self.bn1 = nn.InstanceNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, bias=False))
        self.bn2 = nn.InstanceNorm2d(out_channel)
        self.downsample = downsample
        self.upsample = upsample

    def forward(self, x):
        identity = x
        if self.upsample is not None:
            x = self.upsample(x)
            identity = self.downsample(identity)
        if self.downsample is not None and self.upsample is None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet_seg(nn.Module):

    def __init__(self,
                 block=BasicBlock,

   ):
        super(ResNet_seg, self).__init__()
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(3, 16, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(0.2, inplace=True) )


        # if bilinear, use the normal convolutions to reduce the number of channels
        self.out_seg_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 32, 3),
                                     nn.InstanceNorm2d(32),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(32, 1, 3),
                                    nn.Sigmoid()
                                     )
        self.out_seg = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(1, 32, 3),
                                     nn.InstanceNorm2d(32),
                                     nn.LeakyReLU(0.2, inplace=True))



        self.layer1 = self._make_layer(block, 16,32, 0, stride=2)
        self.layer2 = self._make_layer(block, 32,64,0, stride=2)
        self.layer3 = self._make_layer(block, 64,128,0, stride=2)
        self.layer4 = self._make_layer(block, 128,128, 1, stride=1)
        self.layer9 = self._make_layer(block, 32, 32, 1, stride=1)
        self.layer8 = self._make_layer(block, 160, 128, 3, stride=1)
        self.layer5 = self._make_layer(block, 128,64, 0, stride=2,up=True)
        self.layer6 = self._make_layer(block, 64, 32, 0, stride=2,up=True)
        self.layer7 = self._make_layer(block, 32, 16, 0, stride=2,up=True)
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(16,3 , 7),
                                  nn.Tanh())
        self.merge1=merge(64)
        self.merge2 = merge(32)
        self.merge3 = merge(16)

        self.conv1=nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(128,64,3,1),
                            nn.InstanceNorm2d(64),
                            nn.LeakyReLU(0.2, inplace=True)
                                 )
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(64, 32, 3, 1),
                                   nn.InstanceNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )

        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(32, 16, 3, 1),
                                   nn.InstanceNorm2d(16),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )
        self.drop=nn.Dropout(0.2)
    def _make_layer(self, block, in_channel,channel, block_num, stride=1,up=False,i=0):
        downsample = None
        upsample=None
        if  up==False :
            if stride!=1:
                downsample = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, channel , kernel_size=3, stride=2, bias=False),
                    nn.InstanceNorm2d(channel ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                downsample = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, channel , kernel_size=3, stride=1, bias=False),
                    nn.InstanceNorm2d(channel ),
                    nn.LeakyReLU(0.2, inplace=True))
            layers = []
            layers.append(block(in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            upsample=upsample))
            for i in range( block_num):
                layers.append(block(channel, channel))
        if  up==True:
            upsample=nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2, padding=1, output_padding=1),
                                   nn.InstanceNorm2d(in_channel),
                                   nn.LeakyReLU(0.2, inplace=True))
            downsample = nn.Sequential(
                nn.ConvTranspose2d(in_channel, channel, 3, stride, padding=1, output_padding=1),
                nn.InstanceNorm2d(channel),
            nn.LeakyReLU(0.2, inplace=True))
            layers = []
            layers.append(block(in_channel,
                                channel,
                                downsample=downsample,
                                upsample=upsample))
            for _ in range(i, block_num):
                layers.append(block(channel, channel ))
        return nn.Sequential(*layers)

    def forward(self, x,mode='G'):
        x = self.inc(x)  # [1,16,256,256]
        x1 = self.layer1(x)  # [1,32,128,128]
        x2 = self.layer2(x1)  # [1,64,64,64]
        x3 = self.layer3(x2)  # [1,128,32,32]]
        label_out=self.out_seg_conv(x3)
        if mode=='C':
            return label_out
        label_features=self.out_seg(label_out)
        features=self.layer4(x3)
        label_features=self.layer9(label_features)
        features=torch.cat((label_features,features),1)
        features=self.layer8(features)
        features=self.drop(features)
        x2 = self.merge1(x2, features)#[1,64,64,64]
        x4 = self.layer5(features)  # [1,64,64,64]
        x2=torch.cat((x2,x4),1)#[1,128,64,64]
        x2=self.conv1(x2)#[1,128,32,32]
        x1 = self.merge2(x1, x2)
        x4 = self.layer6(x2)  # [1,64,64,64]
        x1=torch.cat((x1,x4),1)
        x1=self.conv2(x1)
        x = self.merge3(x, x1)
        x4 = self.layer7(x1)  # [1,32,128,128]
        x=torch.cat((x,x4),1)
        x = self.conv3(x)
        x = self.outc(x)
        return x, features,label_out



class merge(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.gate = nn.Sequential(

            nn.Conv2d(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        self.up= nn.Sequential(nn.ConvTranspose2d(in_features*2, in_features, 3, stride=2, padding=1, output_padding=1),
                               nn.InstanceNorm2d(in_features),
                               nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        self.attention=PA_Block(in_features)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x1, x2):
        x1 = self.gate(x1)
        x3=self.up(x2)
        x2 = self.attention(self.relu(x1+x3))
        return x2

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1 ),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])


if __name__ == '__main__':

    ma = ResNet_seg()
    netD_A=Discriminator(3)
    input1 = torch.ones(1, 3, 256,256)

    out,A,out1= ma(input1)
    D=netD_A(out)
    print()


# #