import xarray as xr
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from Training_LDM.Downscaling_Dataset_Prep import DownscalingDataset

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
        self.train_dataset = DownscalingDataset(
            input_ds=xr.open_dataset(self.train_input),
            target_ds=xr.open_dataset(self.train_target),
            config={"variables": self.preprocessing.get("variables", {}),
                    "preprocessing": self.preprocessing.get("preprocessing", {})},
            elevation_path=self.elevation,
        )
        if self.val_input and self.val_target:
            self.val_dataset = DownscalingDataset(
                input_ds=xr.open_dataset(self.val_input),
                target_ds=xr.open_dataset(self.val_target),
                config={"variables": self.preprocessing.get("variables", {}),
                        "preprocessing": self.preprocessing.get("preprocessing", {})},
                elevation_path=self.elevation,
            )
        else:
            self.val_dataset = None
        if self.test_input and self.test_target:
            self.test_dataset = DownscalingDataset(
                input_ds=xr.open_dataset(self.test_input),
                target_ds=xr.open_dataset(self.test_target),
                config={"variables": self.preprocessing.get("variables", {}),
                        "preprocessing": self.preprocessing.get("preprocessing", {})},
                elevation_path=self.elevation,
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return None