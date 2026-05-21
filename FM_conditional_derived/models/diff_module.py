"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers

FULL Credits to torchcfm library source code : https://github.com/atong01/conditional-flow-matching

Usage of base class for CFM adapted- AsthanaSh
"""




import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning import LightningModule
import torchcfm
import torchdiffeq
from contextlib import contextmanager, nullcontext
from torchdiffeq import odeint
from torchcfm import ConditionalFlowMatcher
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
    ):
        super().__init__()
        self.denoiser = denoiser

        self.unet_regr = unet_regr


        self.cfm= ConditionalFlowMatcher(sigma=10e-8)


        self.conditional = (context_encoder is not None)

        
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        #Parameterisation removed : AsthaanSh

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


#Flow matching forward and training steps : in OG space .,.. no residuals : AsthanaSh

    def forward(self, coarse_pred, target, sigma=0.0, context=None):
        batch_size = coarse_pred.shape[0]
        device = coarse_pred.device
        t = torch.rand(batch_size, device=device).view(-1, 1, 1, 1)


        eps = torch.randn_like(coarse_pred) if sigma > 0 else 0.0


        mu_t = t * target + (1 - t) * coarse_pred
        
        x_t = mu_t + sigma * eps

        if self.conditional and context is not None:
            if isinstance(context, torch.Tensor):
                context = [(context, None)]
            elif isinstance(context, list):
                if not (isinstance(context[0], tuple) and len(context[0]) == 2):
                    context = [(c, None) for c in context]
            context = self.context_encoder(context)
        else:
            context = None

        pred = self.denoiser(x_t, t, context=context)

        return self.get_loss(pred, target)




#Changed for paired flow matching,,, no use of OT : AsthanaSh

    def shared_step(self, batch):
        (x, y) = batch  # x: LR input, y: HR target: AsthanaSh,,, no OT pairing, pairing indexwise. 
        assert not torch.any(torch.isnan(x)).item(), 'input data has NaNs'
        assert not torch.any(torch.isnan(y)).item(), 'target has NaNs'

        with torch.no_grad():

            coarse_pred = self.unet_regr(x)  # x0



        t, xt, ut = self.cfm.sample_location_and_conditional_flow(coarse_pred, y)



        context = [(coarse_pred, None)] if self.conditional else None


        if self.conditional and context is not None:


            if isinstance(context, torch.Tensor):
                context = [(context, None)]
            elif isinstance(context, list):
                if not (isinstance(context[0], tuple) and len(context[0]) == 2):
                    context = [(c, None) for c in context]
            context = self.context_encoder(context)
        else:
            context = None

        v_pred = self.denoiser(xt, t.view(-1, 1, 1, 1), context=context)

        return self.get_loss(v_pred, ut)


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



    
    @torch.no_grad()
    def sample(self, x, num_steps=1, use_ema=True, coarse_pred=None):
        if coarse_pred is None:
            coarse_pred = self.unet_regr(x)   # x0

        context = [(coarse_pred, None)] if getattr(self, "conditional", False) else None


        if getattr(self, "conditional", False) and context is not None and hasattr(self, "context_encoder"):

            
            context = self.context_encoder(context)

        def ode_fn(t, xt):
            t_batch = t.expand(xt.shape[0]).view(-1, 1, 1, 1)
            return self.denoiser(xt, t_batch, context=context)

        with self.ema_scope() if use_ema and hasattr(self, "ema_scope") else nullcontext():
            trajectory = odeint(
                ode_fn,
                coarse_pred,                          
                torch.linspace(0, 1, num_steps).to(x.device),  
                method="euler",                       
                atol=1e-4,
                rtol=1e-4,
            )
        return trajectory[-1]