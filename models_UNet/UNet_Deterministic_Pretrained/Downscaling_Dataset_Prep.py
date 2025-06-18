import torch
from torch.utils.data import Dataset
import numpy as np

class DownscalingDataset(Dataset):
    def __init__(self, input_ds, target_ds, config):
        """
        input_ds, target_ds: set of four variables
        config: in the config.yaml file
        """
        input_var_names = list(config["variables"]["input"].values())
        target_var_names = list(config["variables"]["target"].values())

        self.input_vars = [input_ds[var] for var in input_var_names]
        self.target_vars = [target_ds[var] for var in target_var_names]

        self.handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
        self.nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)

        self.length = len(self.input_vars[0].time)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Extracting time slice for each variable
        input_slices = [var.isel(time=index).values for var in self.input_vars]
        target_slices = [var.isel(time=index).values for var in self.target_vars]

        if self.handle_nan:
            input_slices = [np.nan_to_num(arr, nan=self.nan_value) for arr in input_slices]
            target_slices = [np.nan_to_num(arr, nan=self.nan_value) for arr in target_slices]

        input_tensor = torch.tensor(np.stack(input_slices)).float()
        target_tensor = torch.tensor(np.stack(target_slices)).float()

        return input_tensor, target_tensor
