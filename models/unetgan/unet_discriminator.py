import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.unetgan import layers


class Unet_DiscriminatorGenerator(nn.Module):
    def __init__(self, input_nc, output_nc=None, ngf=64, use_dropout=False, resolution=512, gpu_ids=[]):
        super(Unet_DiscriminatorGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.model = Unet_Discriminator(input_nc, resolution=resolution)

    def forward(self, inp):
        if self.gpu_ids and isinstance(inp.data, torch.cuda.FloatTensor):
            return torch.nn.parallel.data_parallel(self.model, inp, self.gpu_ids)
        else:
            return self.model(inp)


def D_unet_arch(input_c=3, ch=64, attention='64', ksize='333333', dilation='111111', out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 16, 8 * n, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 5 + [False] * 5,
                 'upsample': [False] * 5 + [True] * 5,
                 'resolution': [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 11)}}

    arch[256] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 8, 16, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 6 + [False] * 6,
                 'upsample': [False] * 6 + [True] * 6,
                 'resolution': [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 13)}}

    arch[512] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 8, 16, 32, 16*2, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 32, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 7 + [False] * 7,
                 'upsample': [False] * 7 + [True] * 7,
                 'resolution': [256, 128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 15)}}
    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, input_c, D_ch=64, D_wide=True, resolution=512, #512,# 256
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), # nn.LeakyReLU(0.1, inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection=True, **kwargs):
        super(Unet_Discriminator, self).__init__()

        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16

        if self.resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        elif self.resolution == 512:
            self.save_features = [0, 1, 2, 3, 4, 5, 6]
        self.out_channel_multiplier = 1  # 4
        # Architecture
        self.arch = D_unet_arch(input_c, self.ch, self.attention, out_channel_multiplier=self.out_channel_multiplier)[resolution]

        self.unconditional = True  # kwargs["unconditional"]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)

            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv=self.which_conv,
                                               wide=self.D_wide,
                                               activation=self.activation,
                                               preactivation=(index > 0),
                                               downsample=(
                                                   nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (
                    functools.partial(F.interpolate, scale_factor=2, mode="nearest")  # mode=nearest is default
                    if self.arch['upsample'][index] else None)

                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                out_channels=self.arch['out_channels'][index],
                                                which_conv=self.which_conv,
                                                # which_bn=self.which_bn,
                                                activation=self.activation,
                                                upsample=upsample_function, skip_connection=True)]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition:  # index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        #last_layer = nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1)
        last_layer = nn.Sequential(*[nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1), nn.Sigmoid()])
        self.blocks.append(last_layer)
        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        # self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        self.linear_middle = self.which_linear(32 * self.ch if resolution == 512 else 16 * self.ch, output_dim)
        self.sigmoid = nn.Sigmoid()
        # Embedding for projection discrimination
        # if not kwargs["agnostic_unet"] and not kwargs["unconditional"]:
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1]+extra)
        if not self.unconditional:  # kwargs["unconditional"]:
            self.embed_middle = self.which_embedding(self.n_classes, 16 * self.ch)
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        print("_____params______")
        for name, param in self.named_parameters():
            print(name, param.size())

        # Set up optimizer
        # self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        """
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        """
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index == 6:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 7:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 8:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 9:  #
                    h = torch.cat((h, residual_features[1]), dim=1)

            elif self.resolution == 256:
                if index == 7:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 8:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 9:  #
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 10:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[1]), dim=1)

            elif self.resolution == 512:
                if index == 8:
                    h = torch.cat((h, residual_features[6]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 10:  #
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 12:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 13:  #
                    h = torch.cat((h, residual_features[1]), dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index == self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                bottleneck_out = self.sigmoid(bottleneck_out)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0
                else:
                    # this is the bottleneck classifier c
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0
        else:
            emb = self.embed(y)
            emb = emb.view(emb.size(0), emb.size(1), 1, 1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            ################
        out = out + proj

        out = out.view(out.size(0), 1, self.resolution, self.resolution)

        return out, bottleneck_out


"""
m = Unet_Discriminator()
x = torch.rand((23,3,128,128))
y = m(x)
print("ok")
"""
