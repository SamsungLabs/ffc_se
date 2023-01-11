"""Module contains Generator implementation.

Implements proposed generator in two variants: autoencoder and unet.
Also contains dataclasses with model and stft parameters, which are filled
from config when initializing run.

"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_utils import AddSkipConn, ConcatSkipConn, get_padding
from .ffc_modules import FFCResNetBlock
from .models_registry import generators


@dataclass
class ModelConfig:
    """Dataclass with model parameters

    Attributes:
        mode:
            "ae" or "unet"
        special_estimate:
            "phases", "decoupled" or None. Define various additional configurations in
            which model could be run. "phases" was used for ablation experiments with phase prediction.
            "decoupled" tested other ideas with decoupled phase and magnitude predictions.
            By default None.
        out_channels:
            starting channels for 1-st and last conv
        scale_factor:
            determines depth of the model
        block_depth:
            list containing number of ffc blocks at each level.
            E.g. block_depth[-1] refers to number of blocks in bottleneck layer.
            If architecture utilizes FFC blocks not only in bottleneck layer, but also while downsampling/upsampling,
            len(block_depth) == scale_factor + 1.
        use_connection:
            "add" or "concat" – determines the skip-connection between layers in unet architecture. If None,
            becomes autoencoder.
        stride, padding_type, bias:
            default downsampling/upsampling convolution parameters if not specified otherwise.
            padding_type is padding_mode for conv.
        fu_kernel:
            kernel size of convolution in Fourier Unit. Usually it is equal to 1.
        ffc_conv_kernel:
            kernel_size for convolutions in FFC blocks. Usually it is equal to 3.
        ffc_global_ratio_in: input ratio of global channels for FFC
        ffc_global_ratio_out: output ratio of global channels for FFC
        fft_norm: norm used for FFT.
        use_only_freq: determines FFT dimensions: Frequency if True or Frequency + Time if False

    """
    mode: str
    special_estimate: Optional[str]
    out_channels: int
    scale_factor: int
    block_depth: List[int]
    use_connection: Optional[str]
    stride: List[int]
    padding_type: str
    bias: bool
    fu_kernel: int
    ffc_conv_kernel: int
    ffc_global_ratio_in: List[float]
    ffc_global_ratio_out: List[float]
    fft_norm: str
    use_only_freq: bool


@dataclass
class STFTConfig:
    """Dataclass containing STFT params.

    Attributes:
        Refer to https://pytorch.org/docs/stable/generated/torch.stft.html

    """
    n_fft: int
    hop_length: int
    win_length: int
    return_complex: bool
    center: bool
    window: Optional[str]

    def get_window(self):
        window = None

        if self.window == "hann_window":
            window = torch.hann_window(self.win_length, periodic=False)
        elif self.window == "hamming_window":
            window = torch.hamming_window(self.win_length, periodic=False)

        return window


class STFTModelWrapper(torch.nn.Module):
    """Wrapper around generator.

    Model trains to predict denoised STFT representation.
    STFT -> generation -> ISTFT


    """
    def __init__(
        self,
        net,
        stft_config: STFTConfig,
        scale_factor: int = 0,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.net = net
        self.scale_factor = scale_factor
        self.padding_mode = padding_mode
        self.stft_kwargs = asdict(stft_config)
        del self.stft_kwargs["window"]

        window_data = stft_config.get_window()
        self.register_buffer(name="window", tensor=window_data, persistent=False)

    def forward(self, x):

        # (B, 1, T) -> (B, T)
        x = x.squeeze(1)
        original_dim = x.size(-1)

        # (B, T) -> (B, Ch, F, Spec_T)
        x = torch.stft(x, window=self.window, **self.stft_kwargs)

        out = self.net(x)

        # (B, Ch, F, Spec_T) -> (B, T)
        out = torch.istft(
            out, window=self.window, length=original_dim, **self.stft_kwargs
        )

        return out.unsqueeze(1)


class DecoupledSTFTPrediction(torch.nn.Module):
    """Implements decoupled stft prediction variant."""
    def __init__(
        self, model, bounding_func_name: str = "sigmoid", only_pred_phases: bool = False
    ):
        super().__init__()
        self.model = model
        self.bounding_function = self._get_bounding_func(bounding_func_name)
        self.only_pred_phases = only_pred_phases

    @staticmethod
    def _get_magnitude_cosine_sine(a: torch.Tensor):
        assert a.size()[-1] == 2

        real, imaginary = a[..., 0], a[..., 1]

        magnitude = (real * real + imaginary * imaginary).sqrt()
        cosine = real / (magnitude + 1e-9)
        sine = imaginary / (magnitude + 1e-9)

        return magnitude.unsqueeze(-1), cosine, sine

    @staticmethod
    def _get_bounding_func(func_str: str):
        f = None
        if func_str == "softplus":
            f = nn.Softplus()
        elif func_str == "sigmoid":
            f = nn.Sigmoid()
        else:
            raise NotImplementedError()

        return f

    def forward(self, stft_representation):
        spectrogram, X_cos, X_sin = self._get_magnitude_cosine_sine(stft_representation)

        reconstructed_output = self.model(spectrogram)

        if self.only_pred_phases:
            _, phase_cos, phase_sin = self._get_magnitude_cosine_sine(
                reconstructed_output
            )
            S_mag = spectrogram
        else:
            bounded_mag_mask = self.bounding_function(
                reconstructed_output[..., 0]
            ).unsqueeze(-1)
            Q, P = (
                reconstructed_output[..., 1].unsqueeze(-1),
                reconstructed_output[..., 2:],
            )

            _, M_cos, M_sin = self._get_magnitude_cosine_sine(P)
            S_mag = F.relu(bounded_mag_mask * spectrogram + Q)
            phase_cos, phase_sin = (
                X_cos * M_cos - X_sin * M_sin,
                X_cos * M_sin + X_sin * M_cos,
            )

        reconstructed_output = S_mag * torch.stack((phase_cos, phase_sin), dim=-1)

        return reconstructed_output


class UNetBaseBlock(torch.nn.Module):
    """Base UNet building block.

    Contains
    """
    def __init__(
        self,
        start_block=nn.Identity(),
        downsample_block=nn.Identity(),
        net=nn.Identity(),
        upsample_block=nn.Identity(),
        end_block=nn.Identity(),
        use_connection: str = None,
    ):
        super().__init__()

        module_dict = nn.Sequential(
            OrderedDict(
                [
                    ("start_block", start_block),
                    ("downsample", downsample_block),
                    ("internal_block", net),
                    ("upsample", upsample_block),
                    ("end_block", end_block),
                ]
            )
        )

        self.model = self._wrap_connection(
            connection_type=use_connection, module=module_dict
        )

    @staticmethod
    def _wrap_connection(connection_type: str, module):

        if connection_type == "concat":
            module = ConcatSkipConn(module)
        elif connection_type == "add":
            module = AddSkipConn(module)

        return module

    def forward(self, x):
        return self.model(x)


@generators.add_to_registry("ffc_se")
class FFCSE(torch.nn.Module):
    def __init__(
        self,
        model_params: dict,
        stft_params: dict,
    ):
        super().__init__()
        stft_config = STFTConfig(**stft_params)
        model_config = ModelConfig(**model_params)

        extended_channels = model_config.out_channels * (2**model_config.scale_factor)
        model = nn.Sequential(
            *[
                FFCResNetBlock(
                    extended_channels,
                    extended_channels,
                    alpha_in=model_config.ffc_global_ratio_in[-1],
                    alpha_out=model_config.ffc_global_ratio_out[-1],
                    kernel_size=model_config.ffc_conv_kernel,
                    padding_type=model_config.padding_type,
                    bias=model_config.bias,
                    fu_kernel=model_config.fu_kernel,
                    fft_norm=model_config.fft_norm,
                    use_only_freq=model_config.use_only_freq,
                )
                for _ in range(model_config.block_depth[-1])
            ]
        )

        for i in range(model_config.scale_factor, 0, -1):
            cur_ch = model_config.out_channels * (2**i)
            start_block = (
                nn.Sequential(
                    *[
                        FFCResNetBlock(
                            cur_ch // 2,
                            cur_ch // 2,
                            alpha_in=model_config.ffc_global_ratio_in[i - 1],
                            alpha_out=model_config.ffc_global_ratio_out[i - 1],
                            kernel_size=model_config.ffc_conv_kernel,
                            padding_type=model_config.padding_type,
                            bias=model_config.bias,
                            fu_kernel=model_config.fu_kernel,
                            fft_norm=model_config.fft_norm,
                            use_only_freq=model_config.use_only_freq,
                        )
                        for _ in range(model_config.block_depth[i - 1])
                    ]
                )
                if model_config.mode == "unet"
                else nn.Identity()
            )
            end_block = (
                nn.Sequential(
                    *[
                        FFCResNetBlock(
                            cur_ch // 2,
                            cur_ch // 2,
                            alpha_in=model_config.ffc_global_ratio_in[i - 1],
                            alpha_out=model_config.ffc_global_ratio_out[i - 1],
                            kernel_size=model_config.ffc_conv_kernel,
                            padding_type=model_config.padding_type,
                            bias=model_config.bias,
                            fu_kernel=model_config.fu_kernel,
                            fft_norm=model_config.fft_norm,
                            use_only_freq=model_config.use_only_freq,
                        )
                        for _ in range(model_config.block_depth[i - 1])
                    ]
                )
                if model_config.mode == "unet"
                else nn.Identity()
            )
            kernel = 1 if model_config.mode == "unet" else 3
            padding = get_padding(kernel)
            conv = nn.Conv2d(
                cur_ch // 2,
                cur_ch,
                kernel_size=kernel,
                padding=padding,
                padding_mode=model_config.padding_type,
                stride=model_config.stride,
                bias=model_config.bias,
            )
            downsample_block = (
                conv
                if model_config.mode == "unet"
                else nn.Sequential(conv, nn.BatchNorm2d(cur_ch), nn.ReLU(True))
            )
            upsample_block = (
                nn.Sequential(
                    nn.Upsample(scale_factor=tuple(model_config.stride)),
                    nn.Conv2d(
                        cur_ch, cur_ch // 2, kernel_size=kernel, bias=model_config.bias
                    ),
                )
                if model_config.mode == "unet"
                else nn.Sequential(
                    nn.ConvTranspose2d(
                        cur_ch,
                        cur_ch // 2,
                        kernel_size=kernel,
                        stride=model_config.stride,
                        bias=model_config.bias,
                        padding=padding,
                        output_padding=(
                            model_config.stride[0] - 1,
                            model_config.stride[1] - 1,
                        ),
                    ),
                    nn.BatchNorm2d(cur_ch // 2),
                    nn.ReLU(True),
                )
            )
            model = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "UNetBaseBlock",
                            UNetBaseBlock(
                                start_block=start_block,
                                downsample_block=downsample_block,
                                net=model,
                                upsample_block=upsample_block,
                                end_block=end_block,
                                use_connection=model_config.use_connection,
                            ),
                        ),
                        (
                            "ConcatConv",
                            nn.Conv2d(
                                cur_ch,
                                cur_ch // 2,
                                kernel_size=1,
                                bias=model_config.bias,
                            )
                            if model_config.use_connection == "concat"
                            else nn.Identity(),
                        ),
                    ]
                )
            )

        start_channels = 1 if model_config.special_estimate is not None else 2
        end_channels = 4 if model_config.special_estimate == "decoupled" else 2

        start_conv = nn.Conv2d(
            start_channels,
            model_config.out_channels,
            kernel_size=7,
            padding=3,
            padding_mode=model_config.padding_type,
            bias=model_config.bias,
        )
        model = nn.Sequential(
            start_conv
            if model_config.mode == "unet"
            else nn.Sequential(
                start_conv, nn.BatchNorm2d(model_config.out_channels), nn.ReLU(True)
            ),
            model,
            nn.Conv2d(
                model_config.out_channels,
                end_channels,
                kernel_size=7,
                padding=3,
                padding_mode=model_config.padding_type,
                bias=model_config.bias,
            ),
        )

        class _InnerWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            @staticmethod
            def _pad_to_divisible(x, modulo: int = 1, *args, **kwargs):
                """
                Pad two last STFT dimensions to be divisible by modulo
                """

                if kwargs["mode"] == "zeros":
                    kwargs["mode"] = "constant"

                dim_1 = (modulo - x.size(-1) % modulo) % modulo
                dim_2 = (modulo - x.size(-2) % modulo) % modulo
                return F.pad(x, (0, dim_1, 0, dim_2), *args, **kwargs)

            def forward(self, x):
                x = x.permute(0, 3, 1, 2)

                dim_2, dim_1 = x.size()[-2:]
                x = self._pad_to_divisible(
                    x,
                    modulo=2**model_config.scale_factor,
                    mode=model_config.padding_type,
                )

                out = self.net(x)[..., :dim_2, :dim_1].permute(0, 2, 3, 1)
                return out

        model = _InnerWrapper(model)

        if model_config.special_estimate is not None:
            only_pred_phases = (
                True if model_config.special_estimate == "phases" else False
            )
            model = DecoupledSTFTPrediction(model, only_pred_phases=only_pred_phases)

        model = STFTModelWrapper(
            model,
            stft_config=stft_config,
            scale_factor=model_config.scale_factor,
            padding_mode=model_config.padding_type,
        )

        self.model = model

    def forward(self, x):
        return self.model(x)
