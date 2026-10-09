#!/usr/bin/env python3
"""Second fixed-eta candidate; exploratory follow-up, not held-out validation."""
from pathlib import Path
from simulate_pit_microphase import Config,run
from validate_pit_microphase import MEANS

if __name__=='__main__':
    for seed in [7,1,42]:
        run(Config(seed=seed,heterogeneity=.12,steps=2500),MEANS,Path(f'artifacts/pit_microphase/candidate2/seed{seed}'))
