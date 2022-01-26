#!/usr/bin/env python

"""
Module flattens corrmats.npy tensor into a vox x vox matrix.
"""

# Setup -------------------------------------------------------------------------------
# PYTHON 3.9 WITH NILEARN 0.7.1
import os
from os.path import exists
import nibabel as nib
from nilearn.image import resample_img, math_img
from nilearn import input_data
import glob, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
from pathlib import Path

def part02(matlab_dir = r"C:\Program Files\MATLAB\R2020a\bin", ):
  """
  Flattens corrmats.npy tensor into a vox x vox matrix.

  Parameters
  ----------
  matlab_dir : str
    Name of the directory for matlab.

  Returns
  -------
  None

  Notes
  -----
  None

  """
  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')

  # Collapsing Tensor  -----------------------------------------------------------------
  if (not exists(os.path.join(WORKSPACEROOT, 'avgcorrvoxfisher.npy'))):
    corrmats = np.load(os.path.join(WORKSPACEROOT, 'corrmats.npy'))
    sublist = np.load(os.path.join(WORKSPACEROOT, 'sublist.npy'))
    fishercorrmats = np.arctanh(corrmats)
    avgcorr = np.nanmean(fishercorrmats, axis=0)
    np.fill_diagonal(avgcorr,0)
    os.chdir(WORKSPACEROOT)
    with open('avgcorrvoxfisher.npy', 'wb') as f:
      np.save(f, avgcorr)

  # avgcorrvox = np.load(os.path.join(WORKSPACEROOT, 'avgcorrvoxfisher.npy'))
  os.chdir(matlab_dir)
  # os.system(".\\matlab.exe -batch \"cd F:\\+CODE\\+JUPYTER\\resting_ht\\src; htcommdetection('F:\\+CODE\\+JUPYTER\\resting_ht\\results'); quit\"")  
  os.system(f".\\matlab.exe -batch \"cd {cwd}; htcommdetection('{WORKSPACEROOT}'); quit\"")
  os.chdir(cwd)
  os.replace("./commat100run.npy", "../results/commat100run.npy")
  os.replace("./commlvl2.npy", "../results/commlvl2.npy")

if __name__ == "__main__":
  part01()