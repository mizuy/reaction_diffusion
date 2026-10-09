#!/usr/bin/env python3
"""Recorded parameter exploration; separate from the first frozen validation."""
from pathlib import Path
from simulate_pit_microphase import Config,run
from validate_pit_microphase import MEANS

def main():
    for eta,corr in [(.15,3),(.3,3),(.15,6),(.3,6)]:
        run(Config(size=96,steps=1800,heterogeneity=eta,correlation=corr),[-.55,-.35,-.15,-.05,.05,.15,.25],Path(f'artifacts/pit_microphase/search/eta{eta}_corr{corr}'))
    for eta in [.12,.16]:
        run(Config(size=192,heterogeneity=eta,steps=2000),MEANS,Path(f'artifacts/pit_microphase/search/large_eta{eta}'))
if __name__=='__main__':main()
