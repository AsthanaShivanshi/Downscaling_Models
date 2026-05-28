"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers

FULL Credits to torchcfm library from which two FM approaches have been adopted. : https://github.com/atong01/conditional-flow-matching

Usage of base class for CFM adapted- AsthanaSh

Both from Noise and refinement of Unet pred (experimental only). 

Variance Presevring FM added as an option : AsthanaSh
"""

import torch
from lightning import LightningModule
from contextlib import contextmanager, nullcontext


from torchdiffeq import odeint
from torchcfm import ConditionalFlowMatcher, VariancePreservingConditionalFlowMatcher
from FM_conditional_derived.models.components.diff.denoiser.ema import LitEma



class FMContextual(LightningModule):
    def __init__(self,
        denoiser,
        context_encoder=None,
        unet_regr=None,
        lr_warmup=0,
        lr=1e-4,
        loss_type="l2",
        use_ema=True,
        ema_decay=0.9999,
        fm_type="cfm",
        fm_sigma=1e-7, #Training deterministically
        source_init="noise",      # "noise", "coarse", "coarse+noise"
        source_noise_std=1.0,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.unet_regr = unet_regr

        self.fm_type = fm_type.lower()
        self.fm_sigma = fm_sigma
        self.source_init = source_init.lower()
        self.source_noise_std = source_noise_std

        if self.source_init not in {"noise", "encoder+noise"}:
            raise ValueError(
                f"Unsupported source_init='{source_init}'. Expected one of: "
                f"['noise', 'encoder+noise']"
            )

        if self.fm_type == "cfm":
            self.cfm = ConditionalFlowMatcher(sigma=self.fm_sigma)
        elif self.fm_type == "vpfm":
            self.cfm = VariancePreservingConditionalFlowMatcher(sigma=self.fm_sigma)
        else:
            raise ValueError(
                f"Unsupported fm_type='{fm_type}'. Expected one of: "
                f"['cfm', 'vpfm']"
            )

        self.conditional = (context_encoder is not None)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        self.use_ema = use_ema

        if self.use_ema:
            self.denoiser_ema = LitEma(self.denoiser, decay=ema_decay)
        self.loss_type = loss_type



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



    def get_loss(self, pred, target, mean=True):

        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")
        return loss



    def _make_source(self, ref_tensor, coarse_pred=None, noise_std=None):
        std = self.source_noise_std if noise_std is None else noise_std

        if self.source_init == "noise":
            return std * torch.randn_like(ref_tensor)
        



        elif self.source_init == "encoder+noise":
            if coarse_pred is None:
                raise ValueError("coarse_pred is required for source_init='encoder+noise'.")
            

            return coarse_pred + std * torch.randn_like(coarse_pred)
        else:
            raise RuntimeError(f"Unhandled source_init='{self.source_init}'")






    def forward(self, x_lr, target):
        coarse_pred = self.unet_regr(x_lr)
        if self.source_init == "encoder+noise":
            # Only use encoding for source, not as context
            enc = self.context_encoder([(coarse_pred, None)])
            x0 = enc + self.source_noise_std * torch.randn_like(enc)
            context = None
        else:
            x0 = self._make_source(target, coarse_pred=coarse_pred)
            context = self.context_encoder([(coarse_pred, None)]) if self.conditional else None

        t, x_t, u_t = self.cfm.sample_location_and_conditional_flow(x0, target)
        pred = self.denoiser(x_t, t.view(-1, 1, 1, 1), context=context)
        return self.get_loss(pred, u_t)
    



    def shared_step(self, batch):
        x_lr, y = batch
        coarse_pred = self.unet_regr(x_lr)
        if self.source_init == "encoder+noise":
            enc = self.context_encoder([(coarse_pred, None)])
            x0 = enc + self.source_noise_std * torch.randn_like(enc)
            context = None
        else:
            x0 = self._make_source(y, coarse_pred=coarse_pred)
            context = self.context_encoder([(coarse_pred, None)]) if self.conditional else None

        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, y)
        v_pred = self.denoiser(xt, t.view(-1, 1, 1, 1), context=context)
        return self.get_loss(v_pred, ut)





    @torch.no_grad()
    def sample(
        self,
        x_lr_3ch,      # [B, 3, H, W] for unet_regr, [B, 2, H, W] for diffusion
        num_steps=10,
        use_ema=True,
        init_noise_std=None,
        solver="rk4",
    ):
        coarse_pred = self.unet_regr(x_lr_3ch)

        
        # Use only the first 2 channels for diffusion
        x_lr_2ch = x_lr_3ch[:, :2]
        if self.source_init == "encoder+noise":
            enc = self.context_encoder([(coarse_pred, None)])
            std = self.source_noise_std if init_noise_std is None else init_noise_std
            x0 = enc + std * torch.randn_like(enc)
            context = None
        else:
            x0 = self._make_source(x_lr_2ch, coarse_pred=None, noise_std=init_noise_std)
            context = self.context_encoder([(coarse_pred, None)]) if self.conditional else None

        def ode_fn(t, xt):
            t_batch = t.expand(xt.shape[0]).view(-1, 1, 1, 1)
            return self.denoiser(xt, t_batch, context=context)

        t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=x_lr_2ch.device, dtype=x_lr_2ch.dtype)
        with self.ema_scope() if use_ema else nullcontext():
            trajectory = odeint(
                ode_fn,
                x0,
                t_span,
                method=solver,
                atol=1e-4,
                rtol=1e-4,
            )
        return trajectory[-1]

#------------------------------------------------------------------------------------



    def training_step(self, batch, batch_idx):

        loss = self.shared_step(batch)
        self.log("train/loss", loss, sync_dist=True)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val/loss", loss, **log_params, sync_dist=True)
        self.log("val/loss_ema", loss_ema, **log_params, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("test/loss", loss, **log_params, sync_dist=True)
        self.log("test/loss_ema", loss_ema, **log_params, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.denoiser_ema(self.denoiser)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25
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





