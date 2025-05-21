#Splitting schenme: 70:20:10 with moving blocks for each decade starting 1971
#1971-2020: 50 full years with this scheme
#rest of the years ---> 2021-2023 ----> went into training 

#Total split : Train : 38 yeras
#Val : 10 years
#Test : 5 years-----> total 53 years from 1971-2023

import torch
import pandas as pd
import random
from collections import defaultdict
import xarray as xr
import os

def split_by_decade(data, output_path, file_basename):
    assert "time" in data.dims, "Data must have a 'time' dimension."

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    times = pd.to_datetime(data.time.values)
    year_to_indices = defaultdict(list)

    for idx, t in enumerate(times):
        year_to_indices[t.year].append(idx)

    years = sorted(year_to_indices.keys())
    decade_blocks = []

    start_year = 1971
    while start_year + 9 <= 2020:
        block_years = list(range(start_year, start_year + 10))
        if all(y in year_to_indices for y in block_years):
            decade_blocks.append(block_years)
        start_year += 10

    last_block_year = decade_blocks[-1][-1] if decade_blocks else 1980
    remaining_years = [y for y in years if y > last_block_year]

    train_indices, val_indices, test_indices = [], [], []

    for block in decade_blocks:
        valid_splits = []

        # Moving 7-year window
        for train_start in range(0, 4):
            train_years = block[train_start:train_start + 7]
            remaining = [y for y in block if y not in train_years]

            # 2-year val window in remaining 3 years
            for val_start in range(0, 2):
                val_years = remaining[val_start:val_start + 2]
                test_years = [y for y in remaining if y not in val_years]

                valid_splits.append((train_years, val_years, test_years))

        train_years, val_years, test_years = random.choice(valid_splits)

        def gather_indices(years_list):
            indices = sum([year_to_indices[y] for y in years_list], [])
            return torch.tensor(sorted(indices)) if indices else torch.tensor([])  # Chronological order, not shiffled

        train_indices.append(gather_indices(train_years))
        val_indices.append(gather_indices(val_years))
        test_indices.append(gather_indices(test_years))

    # Remining years go into (2021–2023) to training
    for y in remaining_years:
        if y in year_to_indices:
            train_indices.append(torch.tensor(sorted(year_to_indices[y])))  # Chronological order

    train_idx = torch.cat(train_indices)
    val_idx = torch.cat(val_indices)
    test_idx = torch.cat(test_indices)

    train_data = data.isel(time=train_idx)
    val_data = data.isel(time=val_idx)
    test_data = data.isel(time=test_idx)

    os.makedirs(output_path, exist_ok=True)
    train_data.to_netcdf(os.path.join(output_path, f"{file_basename}_train.nc"))
    val_data.to_netcdf(os.path.join(output_path, f"{file_basename}_val.nc"))
    test_data.to_netcdf(os.path.join(output_path, f"{file_basename}_test.nc"))

    return train_data, val_data, test_data
