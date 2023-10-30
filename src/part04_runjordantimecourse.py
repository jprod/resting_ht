#!/usr/bin/env python

"""
Extracts time course files form HT community masks
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

def part04(comm_mask_prefix = "HTcomm_rerun", roi_loc = None, mdld_dataroot = r'G:\7T_resting_state\modeled', gm_name="threshac1.nii.gz"):
  """
  Extracts time course files form HT community masksy.

  Parameters
  ----------
  comm_mask_prefix : str
    Prefix for the masks used in 
  roi_loc : str
    A glob path to grab all ROI files
  mdld_dataroot : str
    A directory for where all the modeled date is housed one layer deep
  gm_name : str
    A full path to the GM mask

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
  if not Path(tcpath).is_dir():
    print(f"Created timecourse")
    os.mkdir(tcpath)

  os.chdir(mdld_dataroot)
  resid = [file for file in glob.glob("**/res4d.nii.gz", recursive=True)]


  for path in resid:
    tcpath2 = os.path.join(tcpath, f'{path[14:19]}_timecourse')
    if not Path(tcpath2).is_dir():
      print(f"Created timecourse\\{path[14:19]}_timecourse")
      os.mkdir(tcpath2)
    extract_timecourse(path[20:27], path[:-12]+gm_name, path, tcpath2, os.path.join(roi_loc, '*.nii.gz'), comm_mask_prefix, gm_method='between', gm_thresh=[.2, 1.])

if __name__ == "__main__":
  prefix = str(sys.argv[1])
  roi_loc = str(sys.argv[2])
  data_root = str(sys.argv[3])
  gm_name = str(sys.argv[4])
  part04(prefix, roi_loc, data_root, gm_name)