import torch.nn as nn
import torch.nn.functional as F
import torch
from Model.cgdr import CorrelationGuidedDeformableRegistration, LeeSpeckleSuppressor
from Model.Sun_Net import SNUNet_ECAM
import functools
from torch.nn.modules.padding import ReplicationPad2d
def weights_init_normal(m):
    if hasattr(m, "reset_flow_heads"):
        m.reset_flow_heads()
        return
    if hasattr(m, "reset_identity"):
        m.reset_identity()
        return
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight"):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        # Initial convolution block

        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)
class ResnetGenerator2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

##############################
#        Discriminator
##############################

class discriminator_block(nn.Module):

    def __init__(self, in_filters, out_filters, normalize=True):
        super(discriminator_block, self).__init__()

        layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
        if normalize:
            self.layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)


       # self._init_weight()

    def forward(self, x):
        return self.layers(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape,
        use_cgdr=True,
        cgdr_max_flow=8.0,
        cgdr_corr_threshold=0.35,
        cgdr_region_mode="correlation",
        cgdr_scatter_threshold=0.55,
        cgdr_adaptive_gate_alpha=0.25,
        cgdr_target_high_ratio=0.30,
        cgdr_min_high_ratio=0.10,
        cgdr_max_high_ratio=0.60,
        cgdr_residual_suppress=0.35,
        cgdr_low_conf_flow_scale=0.55,
        cgdr_residual_conf_temperature=6.0,
        cgdr_mask_alignment_with_valid=True,
        cgdr_change_preserve_strength=0.0,
        cgdr_change_preserve_threshold=0.35,
        cgdr_change_preserve_temperature=10.0,
        cgdr_use_speckle_filter=True,
        cgdr_use_coarse_fine_split=True,
        filter_speckle_for_cd=False,
        speckle_filter_mode=None,
        use_ucef=False,
        ucef_scale=0.5,
        use_racr=False,
        racr_scale=0.2,
        racr_base_suppress=0.05,
    ):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        self.use_cgdr = use_cgdr
        self.use_racr = use_racr
        if speckle_filter_mode is None:
            speckle_filter_mode = "both" if filter_speckle_for_cd else "none"
        if speckle_filter_mode not in ("none", "both", "fixed_only", "moving_only"):
            raise ValueError(f"Unknown speckle_filter_mode: {speckle_filter_mode}")
        self.speckle_filter_mode = speckle_filter_mode
        self.cd_speckle_filter = LeeSpeckleSuppressor(window_size=5) if speckle_filter_mode != "none" else None
        self.cgdr = (
            CorrelationGuidedDeformableRegistration(
                in_channels=channels,
                high_corr_threshold=cgdr_corr_threshold,
                max_coarse_flow=cgdr_max_flow,
                region_mode=cgdr_region_mode,
                scatter_threshold=cgdr_scatter_threshold,
                adaptive_gate_alpha=cgdr_adaptive_gate_alpha,
                target_high_ratio=cgdr_target_high_ratio,
                min_high_ratio=cgdr_min_high_ratio,
                max_high_ratio=cgdr_max_high_ratio,
                residual_suppress=cgdr_residual_suppress,
                low_conf_flow_scale=cgdr_low_conf_flow_scale,
                residual_conf_temperature=cgdr_residual_conf_temperature,
                mask_alignment_with_valid=cgdr_mask_alignment_with_valid,
                change_preserve_strength=cgdr_change_preserve_strength,
                change_preserve_threshold=cgdr_change_preserve_threshold,
                change_preserve_temperature=cgdr_change_preserve_temperature,
                use_speckle_filter=cgdr_use_speckle_filter,
                use_coarse_fine_split=cgdr_use_coarse_fine_split,
            )
            if use_cgdr
            else None
        )
        self.CD_model = SNUNet_ECAM(
            3,
            2,
            use_ucef=use_ucef,
            ucef_scale=ucef_scale,
            use_racr=use_racr,
            racr_scale=racr_scale,
            racr_base_suppress=racr_base_suppress,
        )

        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.D = nn.Conv2d(512, 1, 4, padding=1)

    def set_cgdr_region_mode(self, mode):
        if self.cgdr is not None:
            self.cgdr.region_mode = mode

    def forward(
        self,
        x1,
        x2,
        return_registration=False,
        valid_mask=None,
        return_cd_aux=False,
        return_registration_aux=False,
    ):
        registration_loss = x1.new_zeros(())
        registration_aux = None
        if self.cgdr is not None:
            need_aux = return_registration or return_registration_aux or self.use_racr
            if return_registration:
                try:
                    x2, registration_loss, registration_aux = self.cgdr(
                        x1, x2, return_aux=need_aux, compute_loss=True, valid_mask=valid_mask
                    )
                except TypeError:
                    # Backward-compatible path for older cgdr.py without valid_mask argument.
                    x2, registration_loss, registration_aux = self.cgdr(
                        x1, x2, return_aux=need_aux, compute_loss=True
                    )
            else:
                try:
                    if need_aux:
                        x2, registration_loss, registration_aux = self.cgdr(
                            x1, x2, return_aux=True, compute_loss=False, valid_mask=None
                        )
                    else:
                        x2, registration_loss = self.cgdr(
                            x1, x2, return_aux=False, compute_loss=False, valid_mask=None
                        )
                except TypeError:
                    if need_aux:
                        x2, registration_loss, registration_aux = self.cgdr(
                            x1, x2, return_aux=True, compute_loss=False
                        )
                    else:
                        x2, registration_loss = self.cgdr(
                            x1, x2, return_aux=False, compute_loss=False
                        )
        if self.cd_speckle_filter is not None:
            if self.speckle_filter_mode == "both":
                x1 = self.cd_speckle_filter(x1)
                x2 = self.cd_speckle_filter(x2)
            elif self.speckle_filter_mode == "fixed_only":
                x1 = self.cd_speckle_filter(x1)
            elif self.speckle_filter_mode == "moving_only":
                x2 = self.cd_speckle_filter(x2)
        if return_cd_aux:
            logit1,logit2,cd_pred,cd_aux = self.CD_model(
                x1, x2, registration_aux=registration_aux, return_aux=True
            )
        else:
            logit1,logit2,cd_pred = self.CD_model(x1, x2, registration_aux=registration_aux)
        real_logit = self.D(self.pad(logit1))
        fake_logit = self.D(self.pad(logit2))
        if return_registration and return_cd_aux and return_registration_aux:
            return real_logit,fake_logit,cd_pred,registration_loss,cd_aux,registration_aux
        if return_registration and return_cd_aux:
            return real_logit,fake_logit,cd_pred,registration_loss,cd_aux
        if return_registration and return_registration_aux:
            return real_logit,fake_logit,cd_pred,registration_loss,registration_aux
        if return_registration:
            return real_logit,fake_logit,cd_pred,registration_loss
        if return_cd_aux and return_registration_aux:
            return real_logit,fake_logit,cd_pred,cd_aux,registration_aux
        if return_cd_aux:
            return real_logit,fake_logit,cd_pred,cd_aux
        if return_registration_aux:
            return real_logit,fake_logit,cd_pred,registration_aux
        return real_logit,fake_logit,cd_pred
