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

from lib.extract_roi_from_list import extract_timecourse

def part04(comm_mask_prefix = "HTcomm_rerun", roi_loc = None, mdld_dataroot = 'F:/+DATA/TEAMVIEWERIN/modeled', gm_name="threshac1.nii.gz"):
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

  if roi_loc is None:
    roi_loc = WORKSPACEROOT

  # Make Directory for timecourse output?
  tcpath = os.path.join(WORKSPACEROOT, 'timecourse')
  os.mkdir(tcpath)

  os.chdir(mdld_dataroot)
  resid = [file for file in glob.glob("**/res4d.nii.gz", recursive=True)]

  for path in resid:
    extract_timecourse(path[20:27], path[:-12]+gm_name, path, tcpath, roi_loc, out_label)

if __name__ == "__main__":
  prefix = str(sys.argv[1])
  roi_loc = str(sys.argv[2])
  part04(prefix, roi_loc)