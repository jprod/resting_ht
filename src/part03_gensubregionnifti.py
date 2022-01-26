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

from pathlib import Path

def part03(a_mask = "Hypothalamus.nii.gz", comm_mask_prefix = "HTcomm_rerun"):
  """
  Extracts signal for a given analysis mask and calcuates a total correlation tensor for a set of runs. Saves the result to the results directory.

  Parameters
  ----------
  a_mask : str
    Name of the mask to base this analysis off of. Orig. Hypothalums.nii.gz.
  comm_mask_prefix : str
    Prefix for new subregion masks.

  Returns
  -------
  None (Saves output to "results" directory)
  Output includes 
    {comm_mask_prefix}_whole.nii.gz - full mask with comms encoded as intergers.
    {comm_mask_prefix}_{i}.nii.gz - separate masks for each comm.

  Notes
  -----
  Default arguments are currently set for HT analysis on author's personal computer.

  """

  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')

  # Loading Data
  avgcorrvox = np.load(os.path.join(WORKSPACEROOT, 'avgcorrvoxfisher.npy'))
  comm_assignment = np.load(os.path.join(WORKSPACEROOT, 'commlvl2.npy'))

  # Loading Target ROI masks
  mask_img = nib.load(os.path.join(WORKSPACEROOT, f"bin_resampled{a_mask}"))
  image_data = mask_img.get_fdata()
  image_shape = image_data.shape

  comm_assignment_re = comm_assignment.reshape(1019,)
  mask_buff = image_data.copy()
  mask_buff[mask_buff == 1] = comm_assignment_re
  comm_mask_img = nib.Nifti1Image(mask_buff, mask_img.affine)

  # Whole Masks
  nib.save(comm_mask_img, os.path.join(WORKSPACEROOT, f"{comm_mask_prefix}_whole.nii.gz"))

  for i in np.unique(comm_assignment):
    mask_buff = image_data.copy()
    mask_buff[mask_buff == 1] = (comm_assignment_re == i).astype(int)
    comm_mask_img = nib.Nifti1Image(mask_buff, mask_img.affine)
    nib.save(comm_mask_img,  os.path.join(WORKSPACEROOT, f"{comm_mask_prefix}_{i}.nii.gz"))

if __name__ == "__main__":
  part03()