"""SAMPLING ONLY."""

import torch
from tqdm import tqdm

@torch.no_grad()



def flow_matching_sample(
    model,  
    conditioning=None,
    shape=None,
    steps=2,
    device="cuda",
    verbose=True,
    x_T=None,
    integration="euler",  # Supports "euler" and "heun"
):
    """
    model: Trained flow matching model (expects .denoiser)
    conditioning: context tensor(s) for conditioning (e.g., output of context_encoder)
    shape: shape of the sample to generate (batch, channels, H, W)
    steps: number of integration steps
    device: device to run on
    x_T: Optionally provide initial noise, else random
    integration: "euler" (default) or "heun"
    """

    if x_T is None:
        x = torch.randn(shape, device=device)
    else:
        x = x_T.to(device)

    t_vals = torch.linspace(1, 0, steps, device=device)  # Integrate from t=1 to t=0
    dt = -1.0 / steps



    if verbose:
        iterator = tqdm(range(steps), desc="CFM Sampler")
    else:
        iterator = range(steps)




    for i in iterator:
        t = t_vals[i]
        t_tensor = torch.full((x.shape[0], 1, 1, 1), t, device=device)
        v = model.apply_denoiser(x, t_tensor, cond=conditioning)

        if integration == "euler" or i == steps - 1:
            x = x + v * dt  # Euler step





        elif integration == "heun":
            # Predictor step
            x_pred = x + v * dt
            t_next = t_vals[i + 1]
            t_next_tensor = torch.full((x.shape[0], 1, 1, 1), t_next, device=device)
            v_next = model.apply_denoiser(x_pred, t_next_tensor, cond=conditioning)


            x = x + 0.5 * (v + v_next) * dt




            
        else:
            raise ValueError(f"Unknown integration method: {integration}")

    return x

# Example usage:
# samples = flow_matching_sample(model, conditioning=context, shape=(batch_size, channels, H, W), steps=50, device="cuda", integration="heun")