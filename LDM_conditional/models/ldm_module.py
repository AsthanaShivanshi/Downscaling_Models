"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule
from contextlib import contextmanager
from functools import partial

from .components.ldm.denoiser import LitEma


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)

    elif schedule == "quadratic":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 2
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.cpu().numpy()

#CRPS
def empirical_crps(samples, target):
    # samples: [n_samples, batch, channels, ...]
    # target: [batch, channels, ...]
    n = samples.shape[0]
    term1 = torch.mean(torch.abs(samples - target.unsqueeze(0)), dim=0)
    term2 = 0.5 * torch.mean(torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1)), dim=(0,1))
    return torch.mean(term1 - term2)



def center_crop(x,ref):
    _, _, h, w = x.shape
    _, _, h_ref, w_ref = ref.shape
    cropped_h = (h - h_ref) // 2
    cropped_w = (w - w_ref) // 2
    return x[:, :, cropped_h:cropped_h + h_ref, cropped_w:cropped_w + w_ref]



class LatentDiffusion(LightningModule):
    def __init__(self,
        denoiser,
        autoencoder,
        context_encoder=None,
        timesteps=1000,
        unet_regr=None,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        lr_patience=3,
        lr_factor=0.5,
        parameterization="eps",  # all assuming fixed variance schedules,, changeable from config to v and x0
        ae_load_state_file:str= None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.autoencoder = autoencoder.requires_grad_(False)
        self.unet_regr = unet_regr

        # Only load state_dict if a path is given and autoencoder is not already loaded
        if ae_load_state_file is not None and not hasattr(autoencoder, "_is_loaded"):
            print(f"Loading VAE weights from: {ae_load_state_file}")
            self.autoencoder.load_state_dict(torch.load(ae_load_state_file)["state_dict"])
            self.autoencoder._is_loaded = True  # Mark as loaded to avoid double-load

        self.conditional = (context_encoder is not None)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        
        self.use_ema = use_ema
        if self.use_ema:
            self.denoiser_ema = LitEma(self.denoiser)

        self.register_schedule(
            beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end, 
            cosine_s=cosine_s
        )

        self.loss_type = loss_type

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.denoiser_ema.store(self.denoiser.parameters())
            self.denoiser_ema.copy_to(self.denoiser)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.denoiser_ema.restore(self.denoiser.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_denoiser(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            # Ensure cond is a list of (tensor, t_relative) tuples
            if isinstance(cond, torch.Tensor):
                cond = [(cond, None)]
            elif isinstance(cond, list):
                # If it's a list of tensors, wrap each as (tensor, None)
                if not (isinstance(cond[0], tuple) and len(cond[0]) == 2):
                    cond = [(c, None) for c in cond]
            cond = self.context_encoder(cond)
        else:
            cond = None
        return self.denoiser(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        # R! sqrt_alphas_cumprod goes from 1 to 0 with t from 0 to 1000
        # so sqrt_one_minus_alphas_cumprod goes from 0 to 1 with t from 0 to 1000
        # i.e. x start has greater weight than noise for lower ts while
        # noise has greater weight than x start for higher ts 
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def get_v(self, x, noise, t):
        # R! sqrt_alphas_cumprod goes from 1 to 0 with t from 0 to 1000
        # so sqrt_one_minus_alphas_cumprod goes from 0 to 1 with t from 0 to 1000
        # i.e. x start has greater weight than noise for higher ts while
        # noise has greater weight than x start for lower ts         
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )




    def get_loss(self, pred, target, mean=True, channelwise=False):
            if self.loss_type == 'crps':
                return empirical_crps(pred, target)
            elif self.loss_type == 'l1':
                loss = (target - pred).abs()
                if channelwise:
                    #mean over batch and spatial
                    dims = tuple(i for i in range(loss.ndim) if i != 1)
                    return loss.mean(dim=dims)
                if mean:
                    loss = loss.mean()
            elif self.loss_type == 'l2':
                loss = (target - pred) ** 2
                if channelwise:
                    dims = tuple(i for i in range(loss.ndim) if i != 1)
                    return loss.mean(dim=dims)
                if mean:
                    loss = loss.mean()
            else:
                raise NotImplementedError(f"unknown loss type '{self.loss_type}'")
            return loss





    def p_losses(self, x_start, t, noise=None, context=None, channelwise=False):
            if self.loss_type == 'crps':
                # Not implemented for channelwise
                n_samples = 10  
                samples = []
                for _ in range(n_samples):
                    noise = torch.randn_like(x_start)
                    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                    denoiser_out = self.denoiser(x_noisy, t, context=context)
                    samples.append(denoiser_out)
                samples = torch.stack(samples)
                if self.parameterization == "eps":
                    target = noise
                elif self.parameterization == "x0":
                    target = x_start
                elif self.parameterization == "v":
                    target = self.get_v(x_start, noise, t)
                else:
                    raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
                return self.get_loss(samples, target, mean=True)
            else:
                if noise is None:
                    noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                denoiser_out = self.denoiser(x_noisy, t, context=context)
                if self.parameterization == "eps":
                    target = noise
                elif self.parameterization == "x0":
                    target = x_start
                elif self.parameterization == "v":
                    target = self.get_v(x_start, noise, t)
                else:
                    raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
                return self.get_loss(denoiser_out, target, mean=False, channelwise=channelwise).mean(), \
                    self.get_loss(denoiser_out, target, mean=False, channelwise=True)




    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)
    


#Loss latent space, not interpretable,,, hence, decoded loss being calculated for logging and checkpoint selection

    def shared_step(self, batch):
        (x, y) = batch  # input, target
        assert not torch.any(torch.isnan(x)).item(), 'input data has NaNs'
        assert not torch.any(torch.isnan(y)).item(), 'target has NaNs'
        if self.autoencoder.ae_flag == 'residual':
            residual, _ = self.autoencoder.preprocess_batch([x, y])
            y_latent = self.autoencoder.encode(residual)[0]
        else:
            y_latent = self.autoencoder.encode(y)[0]  

        coarse_pred = self.unet_regr(x)
        context = self.context_encoder([(coarse_pred, None)]) if self.conditional else None

        # LDM denoising in latent space
        loss_latent, channelwise_loss_latent = self(y_latent, context=context, channelwise=True)

        # Decode denoised latent to physical space and compute losses
        with torch.no_grad():
            t = torch.randint(0, self.num_timesteps, (y_latent.shape[0],), device=self.device).long()
            x_noisy = self.q_sample(x_start=y_latent, t=t)
            denoised_latent = self.denoiser(x_noisy, t, context=context)
            print("y shape:", y.shape)
            print("y_latent shape:", y_latent.shape)
            print("denoised_latent shape:", denoised_latent.shape)
            decoded = self.autoencoder.decode(denoised_latent)  # [B, 2, H, W]
            print("decoded shape:", decoded.shape)

            if decoded.shape[2:] != y.shape[2:]: #For calculating loss,,, cropping to match input-target size
                decoded = center_crop(decoded, y)

            # Compute per-channel loss in physical space
            loss_fn = nn.L1Loss() if self.loss_type == "l1" else nn.MSELoss()
            loss_precip = loss_fn(decoded[:, 0], y[:, 0])
            loss_temp = loss_fn(decoded[:, 1], y[:, 1])

        return loss_latent, channelwise_loss_latent, loss_precip, loss_temp


    def training_step(self, batch, batch_idx):
        loss_latent, _, loss_precip, loss_temp = self.shared_step(batch)
        self.log("train/loss_precip", loss_precip, sync_dist=True)
        self.log("train/loss_temp", loss_temp, sync_dist=True)
        return loss_latent  # has to be latent loss for gradient flow and optim

    @torch.no_grad()

    
    def validation_step(self, batch, batch_idx):
        loss_latent, _, loss_precip, loss_temp = self.shared_step(batch)
        with self.ema_scope():
            _, _, loss_precip_ema, loss_temp_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val/loss_precip", loss_precip, **log_params, sync_dist=True)
        self.log("val/loss_temp", loss_temp, **log_params, sync_dist=True)
        self.log("val/loss_precip_ema", loss_precip_ema, **log_params, sync_dist=True)
        self.log("val/loss_temp_ema", loss_temp_ema, **log_params, sync_dist=True)
        self.log("val/loss_ema", 0.5 * (loss_precip_ema + loss_temp_ema), **log_params, sync_dist=True) #Mean of two val ema losses for early stopping : AsthanaSh

    def test_step(self, batch, batch_idx):
        loss_latent, _, loss_precip, loss_temp = self.shared_step(batch)
        with self.ema_scope():
            _, _, loss_precip_ema, loss_temp_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("test/loss_precip", loss_precip, **log_params, sync_dist=True)
        self.log("test/loss_temp", loss_temp, **log_params, sync_dist=True)
        self.log("test/loss_precip_ema", loss_precip_ema, **log_params, sync_dist=True)
        self.log("test/loss_temp_ema", loss_temp_ema, **log_params, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.denoiser_ema(self.denoiser)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
        betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.lr_patience, factor=self.lr_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss_ema",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self, 
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        **kwargs    
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step+1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure,
            **kwargs
        )
