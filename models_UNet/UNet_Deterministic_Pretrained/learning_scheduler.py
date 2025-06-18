
#Great resource on learning rate schedulers : https://medium.com/data-scientists-diary/guide-to-pytorch-learning-rate-scheduling-b5d2a42f56d4

from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau,StepLR

def get_scheduler(name, optimizer, config):
    if name == "CyclicLR":
        return CyclicLR(
            optimizer,
            base_lr=float(config.get("base_lr", 1e-4)),
            max_lr=float(config.get("max_lr", 1e-3)),
            step_size_up=int(config.get("step_size_up", 208)),
            mode=config.get("scheduler_mode", "triangular") #Can be changed to triangular2 in the .yaml file
        )
    if name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get("scheduler_mode", "min"),
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 3)),
            threshold=float(config.get("scheduler_threshold", 1e-4)))

    elif name== "StepLR":
        return StepLR(optimizer,
                  step_size=int(config.get("step_size",2)),
                  gamma=float(config.get("gamma",0.1)))
        
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
