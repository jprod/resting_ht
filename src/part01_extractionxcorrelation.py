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

def part01(a_mask = "Hypothalamus.nii.gz", mdld_dataroot = 'F:/+DATA/TEAMVIEWERIN/modeled', mask_dir = r'F:\+DATA\ABSFULL\atlases\Pauli_MNI152-Nonlin-Asym-2009c\bilateral_rois'):
  """
  Extracts signal for a given analysis mask and calcuates a total correlation tensor for a set of runs. Saves the result to the results directory.

  Parameters
  ----------
  a_mask : str
    Name of the mask to base this analysis off of. Orig. Hypothalums.nii.gz.
  mdld_dataroot : str
    Upper directory for the total resting niftis
  mask_dir : str
    Directory where all of the masks are stored (atlas directory).

  Returns
  -------
  None (Saves output to "results" directory)
  Output includes 
    corrmats.npy - sub x mask_vox x mask_vox tensor for correlations between voxels within mask
    sublist.npy - subject labels for the first axis of the corrmats.npy tensor 

  Notes
  -----
  Default arguments are currently set for HT analysis on author's personal computer.

  """
  # Set the mask 
  ANALYSIS_MASK = a_mask # Mask file name

  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  MODELEDDATAROOT = mdld_dataroot # Location of resting runs
  MASKDIR = mask_dir # Location of mask
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')
  # customised_masks = True
  maskpath = os.path.join(MASKDIR, ANALYSIS_MASK)
  binmaskpath = os.path.join(WORKSPACEROOT, 'bin_resampled' + ANALYSIS_MASK)

  # Getting File Paths
  os.chdir(MODELEDDATAROOT)
  resid = [file for file in glob.glob("**/res4d.nii.gz", recursive=True)]
  print(resid)

  # Loading, Resampling, and Binarizing Hypothalamus Mask -------------------------------
  resid_img = nib.load(os.path.join(MODELEDDATAROOT, resid[0]))
  mask_img = nib.load(maskpath)
  resampled_mask = resample_img(
      mask_img, target_shape=resid_img.shape[:3], target_affine=resid_img.affine, 
      interpolation='linear') # use linear neighbor interpolation for probabilistic images
  resampled_mask = math_img('img > 0.5', img=resampled_mask)
  nib.save(resampled_mask, binmaskpath)
  masker = input_data.NiftiMasker(mask_img=resampled_mask)
  masker.fit(resid_img)
  del resid_img

  # Process Subject and Runs Loop -------------------------------------------------------
  def processid(res4dpath):
    """Proecesses a res4d pathname into a np matrix of shape (voxels, timepoints)"""
    print("\t\tmasking...")
    # Applying mask
    _resid = nib.load(os.path.join(MODELEDDATAROOT, res4dpath))
    masked_data = masker.transform(_resid)
    del _resid
    if masked_data.shape[1] != np.count_nonzero(resampled_mask.dataobj):
      raise ValueError(f"Num Voxels in Mask, {masked_data.shape[1]}, \
      does not equal non_zero count in the mask, {np.count_nonzero(resampled_mask.dataobj)}")

    # Elbow threshold to remove high varience voxels
    print("\t\tfiltering...")
    voxels = np.transpose(masked_data)
    voxvar = np.var(voxels, axis=1)
    np.sort(voxvar)
    pfit = np.polyfit(x=range(voxvar.shape[0]),y=np.sort(voxvar), deg=2)
    poly = np.poly1d(pfit)
    kneedle = KneeLocator(x=range(voxvar.shape[0]),y=np.sort(voxvar), S=1.0, curve="convex", direction="increasing")
    elbowpoint = round(kneedle.elbow, 3)
    elbowthresh = poly(elbowpoint)
    voxels2 = htvoxels.copy()
    voxels2[voxvar > elbowthresh] = np.repeat(np.nan, voxels.shape[1])
    return voxels2

  BOOLLIST = [[[False]], [[False, False], [True, False], [False, True]],
    [[False, False, False],
     [True, False, False],
     [False, True, False],
     [False, False, True],
     [True, True, False],
     [True, False, True],
     [False, True, True]],]

  def piececorr(listofvox, boolarray):
    """Generate multiple corellations via boolean masks"""
    voxcat = np.concatenate([vox for vox, bool in zip(listofvox, boolarray) if not bool], 1)
    return np.corrcoef(voxcat)

  def corrfill(corr1, corr2):
    """Fill correlations, designed for a reduce operation"""
    fill = corr2[np.isnan(corr1)]
    corr1[np.isnan(corr1)] = fill
    return corr1

  def numnanrows(corr):
   return int(corr.shape[0] - (corr.shape[0]**2 - np.count_nonzero(np.isnan(corr)))**(1/2))

  def procsub(listofres4dpaths):
    """Proecesses a list of res4d pathname into a np matrix of the pearson's r over time"""
    print("\tprocessing subject...")
    start = time.process_time()
    voxlist = [processid(res4dpath) for res4dpath in listofres4dpaths]
    corrs = [piececorr(voxlist, b) for b in BOOLLIST[len(voxlist)-1]]
    totalcorr = reduce(corrfill, corrs)
    np.fill_diagonal(totalcorr,np.nan)
    end = time.process_time()
    print(f"\t\t{end - start} secs")
    print(f"\t\t{numnanrows(totalcorr)} nan of {totalcorr.shape[0]}")
    return totalcorr
          
  def procsubname(subname):
    """Proecesses a subject name (eg. sub-011) into a np matrix of the pearson's r over time"""
    print(subname, end =": \n")
    # Indicies for the file name diffrences (subject name) may differ for naming scheme and total path
    subruns = [p for p in resid if p[20:27] == subname] 
    corrmat = procsub(subruns)
    print("")
    return corrmat

  # Loop blocks using previously defined fns to process all residual niftis
  # Indicies for the file name diffrences (subject name) may differ for naming scheme and total path
  subset =  sorted(set([path[20:27] for path in resid]))
  print(subset)
  print("--START PROCESS--")
  corrmatlist = [procsubname(n) for n in subset]
  output = np.stack(corrmatlist)

  os.chdir(WORKSPACEROOT)
  with open('corrmats.npy', 'wb') as f:
    np.save(f, output)
  with open('sublist.npy', 'wb') as f:
    np.save(f, subset)


if __name__ == "__main__":
  part01()