from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

def normalization(channels, norm_type="group", **norm_kwargs):
    print(f"normalization called with channels={channels}, norm_type={norm_type}, norm_kwargs={norm_kwargs}")
    norm_type = norm_type.lower() if norm_type else "none"
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    elif norm_type == "group":
        num_groups = norm_kwargs.get("num_groups", 8)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)


def activation(act_type="swish"):
    if act_type == "swish":
        return nn.SiLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif not act_type:
        return nn.Identity()
    else:
        raise NotImplementedError(act_type)


class ResBlock2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3,3), 
        act='swish', norm='group', norm_kwargs=None, 
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
        padding = tuple(k//2 for k in kernel_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
            kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)

        self.sequence = nn.Sequential(
            self.norm1,
            self.act1,
            self.conv1,
            self.norm2,
            self.act2,
            self.conv2,
        )

    def forward(self, x):
        return self.sequence(x) + self.proj(x)
    