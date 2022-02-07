#!/usr/bin/env python

"""
Module extracts signal for a given analysis mask and calcuates a total correlation tensor
(sub x mask_vox x mask_vox) for a set of runs
"""

# Setup -------------------------------------------------------------------------------
# PYTHON 3.9 WITH NILEARN 0.7.1
import os
import nibabel as nib
from nilearn.image import resample_img, math_img
from nilearn import input_data
import glob, os
import numpy as np
from kneed import KneeLocator
import time
from functools import reduce

import sys
from pathlib import Path

def part04(comm_mask_prefix = "HTcomm_rerun"):
  """
  Extracts signal for a given analysis mask and calcuates a total correlation tensor for a set of runs. Saves the result to the results directory.

  Parameters
  ----------

  Returns
  -------
  None (Saves output to "results" directory)
  Output includes 
    Timecourse data - timecourse of new masks.

  Notes
  -----
  Default arguments are currently set for HT analysis on author's personal computer.

  """
  print("--Running part 04--")
  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')

  # Make Directory for timecourse output?
  tcpath = os.path.join(WORKSPACEROOT, 'timecourse')
  tc_out = os.mkdir(tcpath)
  raise ValueError("Jordan's script not specified")


if __name__ == "__main__":
  part04()