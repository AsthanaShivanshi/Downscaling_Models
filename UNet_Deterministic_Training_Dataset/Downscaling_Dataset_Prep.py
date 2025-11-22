import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import skimage.transform

class DownscalingDataset(Dataset):
    def __init__(self, input_ds, target_ds, config, elevation_path=None):
        """
        input_ds, target_ds: set of four variables
        config: in the config.yaml file
        """
        input_var_names = list(config["variables"]["input"].values())
        input_channel_names = input_var_names.copy()
        if elevation_path is not None:
            input_channel_names.append("elevation")

        # Printing channel index mapping (debugging)
        #for i, name in enumerate(input_channel_names):
        #    print(f"Input channel {i}: {name}")

        self.input_vars = [input_ds[var] for var in input_var_names]
        self.length = len(self.input_vars[0].time)

        # Only set target_vars if target_ds is not None
        self.target_vars = None
        if target_ds is not None:
            target_var_names = list(config["variables"]["target"].values())
            #for i, name in enumerate(target_var_names):
            #    print(f"Output channel {i}: {name}")
            self.target_vars = [target_ds[var] for var in target_var_names]

        self.handle_nan = config.get("preprocessing", {}).get("nan_to_num", True)
        self.nan_value = config.get("preprocessing", {}).get("nan_value", 0.0)

        # Loading elevation 
        self.elevation = None
        if elevation_path is not None:
            try:
                with rasterio.open(elevation_path) as src:
                    elev = src.read(1)
                    self.elevation = elev.astype(np.float32)
                print(f"Loaded elevation from {elevation_path}, shape: {self.elevation.shape}")
            except Exception as e:
                print(f"Could not load elevation: {e}")
        else:
            print("No elevation path provided, elevation will not be used.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Extracting time slice for each variable
        input_slices = [var.isel(time=index).values for var in self.input_vars]
        if self.handle_nan:
            input_slices = [np.nan_to_num(arr, nan=self.nan_value) for arr in input_slices]

        if self.elevation is not None:
            elev = self.elevation
            if elev.shape != input_slices[0].shape:
                from skimage.transform import resize
                elev = resize(elev, input_slices[0].shape, order=1, preserve_range=True, anti_aliasing=True)
            input_slices.append(elev.astype(np.float32))

        input_tensor = torch.tensor(np.stack(input_slices)).float()

        # Only return target if available
        if self.target_vars is not None:
            target_slices = [var.isel(time=index).values for var in self.target_vars]
            if self.handle_nan:
                target_slices = [np.nan_to_num(arr, nan=self.nan_value) for arr in target_slices]
            target_tensor = torch.tensor(np.stack(target_slices)).float()
            #print(f"Input tensor shape: {input_tensor.shape}, Target tensor shape: {target_tensor.shape}")
            return input_tensor, target_tensor
        else:
            #print(f"Input tensor shape: {input_tensor.shape}")
            return input_tensor