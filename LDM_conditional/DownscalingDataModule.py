import xarray as xr
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from LDM_conditional.Downscaling_Dataset_Prep import DownscalingDataset
import rasterio
import numpy as np

class DownscalingDataModule(LightningDataModule):
    def __init__(
        self,
        train_input,
        train_target,
        val_input=None,
        val_target=None,
        test_input=None,
        test_target=None,
        elevation=None,
        batch_size=32,
        num_workers=4,
        preprocessing=None,
    ):
        super().__init__()
        self.train_input = train_input
        self.train_target = train_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target
        self.elevation = elevation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing = preprocessing or {}

    def setup(self, stage=None):
        # Load elevation once as array
        if self.elevation is not None and isinstance(self.elevation, str):
            with rasterio.open(self.elevation) as src:
                elev = src.read(1).astype(np.float32)
            elevation_array = elev
        else:
            elevation_array = self.elevation

        # Only create train dataset if both dicts are non-empty
        if self.train_input and self.train_target:
            train_input_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.train_input.items()}
            train_target_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.train_target.items()}
            self.train_dataset = DownscalingDataset(
                input_ds=train_input_ds,
                target_ds=train_target_ds,
                config={"variables": self.preprocessing.get("variables", {}),
                        "preprocessing": self.preprocessing.get("preprocessing", {})},
                elevation_path=elevation_array,
            )
        else:
            self.train_dataset = None

        # Val
        if self.val_input and self.val_target:
            val_input_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.val_input.items()}
            val_target_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.val_target.items()}
            self.val_dataset = DownscalingDataset(
                input_ds=val_input_ds,
                target_ds=val_target_ds,
                config={"variables": self.preprocessing.get("variables", {}),
                        "preprocessing": self.preprocessing.get("preprocessing", {})},
                elevation_path=elevation_array,
            )
        else:
            self.val_dataset = None

        # Test
        if self.test_input and self.test_target:
            test_input_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.test_input.items()}
            test_target_ds = {k: xr.open_dataset(v, engine='netcdf4') for k, v in self.test_target.items()}
            self.test_dataset = DownscalingDataset(
                input_ds=test_input_ds,
                target_ds=test_target_ds,
                config={"variables": self.preprocessing.get("variables", {}),
                        "preprocessing": self.preprocessing.get("preprocessing", {})},
                elevation_path=elevation_array,
            )
        else:
            self.test_dataset = None


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, 
                              batch_size=self.batch_size, 
                              num_workers=self.num_workers,
                              persistent_workers=True)
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                persistent_workers=(self.num_workers > 0)
            )
        return None