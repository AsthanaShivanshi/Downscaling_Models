import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize

class DownscalingDataset(Dataset):
    def __init__(self, input_ds, target_ds, config, elevation_path=None):
        input_var_names = list(config["variables"]["input"].keys())
        target_var_names = list(config["variables"]["target"].keys())

        input_channel_names = input_var_names.copy()
        if elevation_path is not None:
            input_channel_names.append("elevation")

        self.input_vars = [
            input_ds[var][config["variables"]["input"][var]] for var in input_var_names
        ]
        self.target_vars = [
            target_ds[var][config["variables"]["target"][var]] for var in target_var_names
        ]

        self.handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
        self.nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)
        self.length = len(self.input_vars[0].time)

        self.elevation = None
        if elevation_path is not None:
            if isinstance(elevation_path, np.ndarray):
                self.elevation = elevation_path.astype(np.float32)
            else:
                print("Elevation provided is not a numpy array, not used.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input_slices = [var.isel(time=index).values for var in self.input_vars]
        target_slices = [var.isel(time=index).values for var in self.target_vars]

        if self.handle_nan:
            target_slices = [np.nan_to_num(arr, nan=self.nan_value).astype(np.float32) for arr in target_slices]
            input_slices = [np.nan_to_num(arr, nan=self.nan_value).astype(np.float32) for arr in input_slices]

        if self.elevation is not None:
            elev = self.elevation
            if elev.shape != input_slices[0].shape:
                elev = resize(elev, input_slices[0].shape, order=1, preserve_range=True, anti_aliasing=True)
            input_slices.append(elev.astype(np.float32))

        # Defensive : debugging not working despite shape being correct : AsthanaSh
        for i, arr in enumerate(input_slices):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"input_slices[{i}] is not a numpy array!")
            if arr.dtype != np.float32:
                raise TypeError(f"input_slices[{i}] is not float32!")
            if arr.shape != input_slices[0].shape:
                raise ValueError(f"input_slices[{i}] shape mismatch: {arr.shape} vs {input_slices[0].shape}")
            if np.any(np.isnan(arr)):
                raise ValueError(f"input_slices[{i}] contains NaNs!")
            if np.any(np.isinf(arr)):
                raise ValueError(f"input_slices[{i}] contains Infs!")

        stacked_input = np.stack(input_slices).astype(np.float32)
        stacked_target = np.stack(target_slices).astype(np.float32)
        if stacked_input.dtype == object:
            raise ValueError("Stacked input_slices is dtype object!")
        if stacked_target.dtype == object:
            raise ValueError("Stacked target_slices is dtype object!")

        input_tensor = torch.from_numpy(stacked_input)
        target_tensor = torch.from_numpy(stacked_target)
        return input_tensor, target_tensor