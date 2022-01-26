#!/usr/bin/env python

"""
Runs all analysis with the parameters below.
"""

import numpy as np
import pandas as pd
import os
from os.path import exists
from part01_extractionxcorrelation import part01
from part02_genmaskcomms import part02
from part03_gensubregionnifti import part03
from part04_runjordantimecourse import part04
from part05_TCcorrandisolation import part05

run_full_override = False # setable flag
DEFAULT_NO_OVERRIDE = False
SINGLE_OVERRIDE_RUN = True
FORCESKIP = False
DEFAULTRUN = True
a_mask = "Hypothalamus.nii.gz"
mdld_dataroot = 'F:/+DATA/TEAMVIEWERIN/modeled'
mask_dir = r'F:\+DATA\ABSFULL\atlases\Pauli_MNI152-Nonlin-Asym-2009c\bilateral_rois'
atlas_dir = "NOP"
matlab_dir = r"C:\Program Files\MATLAB\R2020a\bin"
comm_mask_prefix = "HTcomm_rerun"
timecourse_root = r"F:\+DATA\ABSFULL\timecourse"

def main():
  # EXTRACTION OF MASK VOXELS AND CROSS CORRELATION
  if FORCESKIP and ((not exists(os.path.join('../results/', 'corrmats.npy'))) or run_full_override or DEFAULT_NO_OVERRIDE):
    part01(a_mask, mdld_dataroot, mask_dir)

  # COMMUNITY GENERATION FOR SPECIFIED ROI
  if DEFAULTRUN and ((not exists(os.path.join('../results/', 'commlvl2.npy'))) or run_full_override or DEFAULT_NO_OVERRIDE):
    part02(matlab_dir)

  # MASK CREATION FOR NEWLY GENERATED COMMUNITY ASSIGNMENTS
  if DEFAULTRUN and ((not exists(os.path.join('../results/', f"{comm_mask_prefix}_whole.nii.gz"))) or run_full_override or DEFAULT_NO_OVERRIDE):
    part03(a_mask, comm_mask_prefix)

  # RUN JORDAN'S EXRACTION SCRIPT ON THE NEW SUBREGION MASKS
  if FORCESKIP and ((not exists('../results/timecourse')) or run_full_override or DEFAULT_NO_OVERRIDE):
    part04(comm_mask_prefix)

  # CORRELATE AND AGGREGATE TIMECOURSE DATA
  if DEFAULTRUN and ((not exists(os.path.join('../results/', 'P05_fishercrosssubtotalTCcorr.npy'))) or run_full_override or DEFAULT_NO_OVERRIDE):
    part05(timecourse_root)

  print("--- ANALYSIS FINISH ---")

if __name__ == "__main__":
  main()