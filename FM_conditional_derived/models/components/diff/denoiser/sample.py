"""SAMPLING ONLY."""

import torch
from tqdm import tqdm

@torch.no_grad()


def flow_matching_sample(
    model,  
    context,
    shape,
    steps=2,
    device="cuda",
    verbose=True,
    x_T=None,
    integration="euler",  # Only Euler for now
):
    """
    model: Trained flow matching model (expects .denoiser)
    context: context tensor(s) for conditioning (e.g., output of context_encoder)
    shape: shape of the sample to generate (batch, channels, H, W)
    steps: number of integration steps
    device: device to run on
    x_T: Optionally provide initial noise, else random
    integration: "euler" (default)
    """
    if x_T is None:
        x = torch.randn(shape, device=device)
    else:
        x = x_T.to(device)
    t_vals = torch.linspace(1, 0, steps, device=device)  # Integrate from t=1 to t=0
    dt = -1.0 / steps

    if verbose:
        iterator = tqdm(t_vals, desc="Flow Matching Sampler")


    else:
        iterator = t_vals



    for t in iterator:
        t_tensor = torch.full((x.shape[0], 1, 1, 1), t, device=device)
        v = model.denoiser(x, t_tensor, context=context)
        x = x + v * dt  # Euler step

    return x



# Use like:
# samples = flow_matching_sample(model, context, (batch_size, channels, H, W), steps=2, device="cuda")