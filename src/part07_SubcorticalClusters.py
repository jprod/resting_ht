#!/usr/bin/env python

"""
Module extracts signal for a given analysis mask and calcuates a total correlation tensor
(sub x mask_vox x mask_vox) for a set of runs
"""

# Setup -------------------------------------------------------------------------------
# PYTHON 3.9 WITH NILEARN 0.7.1
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# PYTHON 3.9 WITH NILEARN 0.7.1
import os
import nibabel as nib
from nilearn.image import resample_img, math_img
from nilearn import input_data
import glob, os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
from functools import reduce
from scipy.stats import sem
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path

def part06(TOTALMASKROOT = r"F:\+DATA\ABSFULL\TOTAL ATLAS"):
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
  print("--Running part 06--")
  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')


  meancorr_in = np.load(os.path.join(WORKSPACEROOT, 'P05_fishercrosssubtotalTCcorr.npy'))
  print(f"meancorr_in shape: {meancorr_in.shape}")
  ROIORDER = np.load(os.path.join(WORKSPACEROOT, 'P05_ROIorder.npy'))
  print(f"ROIORDER shape: {ROIORDER.shape}")

  # regex for cortical
  import re
  r = re.compile('[lr]h\.[RL]_.*')
  # TODO: Edit so it uses non HT indicies
  HTindicies = np.where(np.isin(ROIORDER, ["HTcomm_1", "HTcomm_2", "HTcomm_3", "HTcomm_4"]))[0]
  subcortex = np.vectorize(lambda name: not re.match(r, name))(ROIORDER)

  r2 = re.compile('.*Diencephalon-Thal_Hythal.*')
  sum([1 if re.match(r2, name) else 0 for name in ROIORDER])
  WagerHTdicephindicies = np.where(np.vectorize(lambda name: re.match(r2, name))(ROIORDER))
  r3 = re.compile('.*Bstem_PAG.*')
  sum([1 if re.match(r3, name) else 0 for name in ROIORDER])
  WagerPAGindex = np.where(np.vectorize(lambda name: re.match(r3, name))(ROIORDER))
  r4a = re.compile('.*[ap]HIPP.*')
  sum([1 if re.match(r4a, name) else 0 for name in ROIORDER])
  hippAjay = np.where(np.vectorize(lambda name: re.match(r4a, name))(ROIORDER))

  r4y = re.compile('.*_hippocampus_cluster_.*')
  sum([1 if re.match(r4y, name) else 0 for name in ROIORDER])
  hippYuta = np.where(np.vectorize(lambda name: re.match(r4y, name))(ROIORDER))
  r4w = re.compile('.*Wager_atlas-Hippocampus-.*_Hippocampus.*') # Wager_atlas-Hippocampus-CA1_Hippocampus__L
  sum([1 if re.match(r4w, name) else 0 for name in ROIORDER])
  hippWag = np.where(np.vectorize(lambda name: re.match(r4w, name))(ROIORDER))
  r5p = re.compile('.*Wager_atlas.*Bstem_Pons.*') # Wager_atlas-Hippocampus-CA1_Hippocampus__L
  bstempons = np.where(np.vectorize(lambda name: re.match(r5p, name))(ROIORDER))
  r5m = re.compile('.*Wager_atlas.*Bstem_M[ie]d.*') # Wager_atlas-Hippocampus-CA1_Hippocampus__L
  bstemmd = np.where(np.vectorize(lambda name: re.match(r5m, name))(ROIORDER))
  r5th = re.compile('.*Wager_atlas.*Thal_Hb_.*') # Wager_atlas-Hippocampus-CA1_Hippocampus__L
  thalhb = np.where(np.vectorize(lambda name: re.match(r5th, name))(ROIORDER))
  MODE = "ajay"
  subcortex = np.vectorize(lambda name: not re.match(r, name))(ROIORDER)
  subcortex[:7] = False #Inital PAG ROIs
  subcortex[376] = False #HT Pauli L
  subcortex[393] = False #HT Pauli R
  subcortex[-24:-21] = False #Mid PAG ROIs
  subcortex[WagerHTdicephindicies] = False #Wager HT Indicies
  subcortex[WagerPAGindex] = False #Wager PAG Indicies
  subcortex[hippWag] = False #Wager Hipp Indicies
  subcortex[bstempons] = False #Shang Pons
  subcortex[bstemmd] = False #Shang Bstem M[ei]d
  subcortex[thalhb] = False #Small Thal rois
  if MODE == "yuta":
    subcortex[hippAjay] = False # hippocampus ajay is false
  elif MODE == "ajay":
    subcortex[hippYuta] = False

  HTindicies_nocortex = np.where(np.isin(ROIORDER[subcortex], ["HTcomm_1", "HTcomm_2", "HTcomm_3", "HTcomm_4"]))[0]
  subcortex_noHT = subcortex.copy()
  subcortex_noHT[HTindicies] = False
  fcorrisolated = fcorrnp[:, HTindicies, :]
  fcorrisolated = fcorrisolated[:,:,subcortex_noHT]
  fcorriso_moved = np.moveaxis(fcorrisolated, 0, -1)

  meancorr = fcorrisolated
  meancorr_moved = fcorriso_moved

  # calhar_scores = [metrics.calinski_harabasz_score(meancorr_moved, KMeans(n_clusters=i, random_state=1).fit(meancorr_moved).labels_) for i in range(2,10)]\
  distortions = []
  K = range(1,40)
  for k in K:
      kmeanModel = KMeans(n_clusters=k)
      kmeanModel.fit(meancorr_moved)
      distortions.append(kmeanModel.inertia_)
  from kneed import KneeLocator
  kneedle = KneeLocator(K, distortions, S=1.0, curve="convex", direction="decreasing")
  print(kneedle.knee)
  print(kneedle.elbow)

  kmeans_model = KMeans(n_clusters=kneedle.knee, random_state=1).fit(meancorr_moved)
  print(kmeans_model.labels_)
  for i in range(max(kmeans_model.labels_)+1):
    print(f"{i}: {sum(kmeans_model.labels_ == i)}")
  for i in range(kneedle.knee):
    print(f"{i}")
    print(ROIORDER2[kmeans_model.labels_ == i])

  def avg_cluster(clusterset_x_numroi):
    cluster_n = np.max(clusterset_x_numroi)+1
    avgcluster = np.stack([np.nanmean(fcorriso_moved2[clusterset_x_numroi == i], axis=0) for i in range(cluster_n)])
    return(avgcluster)
  corrcluster = avg_cluster(kmeans_model.labels_)

  from statsmodels.stats.anova import AnovaRM
  def getdfanovacluster(clustxsubxht_matrix, savepath="out.csv"):
    nsub = clustxsubxht_matrix.shape[1]
    nclust = clustxsubxht_matrix.shape[0]
    subxht_matrix = np.concatenate([subxhtmat for subxhtmat in clustxsubxht_matrix], axis=1)
    subid = np.arange(nsub).reshape((-1,1))
    full = np.concatenate([subid, subxht_matrix], axis=1)
    print(full)
    print(full.shape)
    long = np.concatenate([np.concatenate([full[:,[0,i]], np.full((nsub,1), (i-1)%4+1),
                                           np.full((nsub,1), (i-1)//4+1)], axis=1) for i in range(1, full.shape[1])], axis=0)
    print(long)
    print(long.shape)
    dflong = pd.DataFrame(long, columns=["pid", "rvalue", "ht_subregion", "cluster"])
  #   dflong.to_csv(savepath)
    dffull = pd.DataFrame(full, columns=["pid"]+[f"rvalue_clust{i//4 + 1}_ht{i%4 + 1}" for i in range(24)])
    dffull.to_csv(savepath, index=False)
    aovrm = AnovaRM(dflong, depvar='rvalue', subject='pid', within=['ht_subregion', 'cluster'])
    res = aovrm.fit()
    print(res)
    return(np.stack([res.anova_table["F Value"].to_numpy(), res.anova_table["Pr > F"].to_numpy()], axis=0))

  anovares2 = getdfanovacluster(corr2cluster, os.path.join(WORKSPACEROOT, "P07subcort_clustersWIDE.csv"))

  os.chdir(TOTALMASKROOT)
  maskpaths = [file for file in glob.glob("**/*.nii.gz", recursive=True)]
  # print(maskpaths)
  def genmasks(row, savename, maskorder=ROIORDER2):
    """Must have WORKSPACEROOT, TOTALMASKROOT and ROILIST defined."""
    os.chdir(TOTALMASKROOT)
    display(row)
    mask = nib.load(f"HTcomm_1.nii.gz")
    maskdata = np.zeros((176, 208, 176))
    for i, e in enumerate(row):
      looproi = f"{maskorder[i]}"
      if looproi[-6:] != 'nii.gz':
        if looproi in ['dorsomedial', 'L-dorsal', 'L-lateral', 'L-ventral', 'R-dorsal', 'R-lateral', 'R-ventral']:
          looproi = f"PAG_clms_{looproi}.nii.gz"
        else:
          looproi = f"PAG_clms-ros-med-cau_{looproi}.nii.gz"
      elif looproi in ['aHIPP_L_ROI.nii.gz', 'aHIPP_R_ROI.nii.gz', 'pHIPP_L_ROI.nii.gz', 'pHIPP_R_ROI.nii.gz']:
          looproi = looproi[:-3]
      print(looproi, end="")
      try:
        roimask = nib.load(looproi)
        print(" - loaded")
        if looproi[2] == '.' or looproi[0] == 'W' or looproi[1:3] == 'HI': 
          roiresampled = resample_img(roimask, target_shape=mask.shape[:3], target_affine=mask.affine, 
            interpolation='nearest') # use linear neighbor interpolation for probabilistic images
          roiresampled = math_img('img > 0.5', img=roiresampled)
  #         view = plotting.view_img(roiresampled, threshold=0)
          roidata = roiresampled.get_fdata()
        elif looproi[1] == '_':      
          roiresampled = resample_img(roimask, target_shape=mask.shape[:3], target_affine=mask.affine, 
            interpolation='linear') # use linear neighbor interpolation for probabilistic images
          roiresampled = math_img('img > 0.5', img=roiresampled)
  #         view = plotting.view_img(roiresampled, threshold=0)
          roidata = roiresampled.get_fdata()
        else:
  #         print(roimask)
          roidata = roimask.get_fdata()
        roidata_weighted = e*roidata
  #       display(view)
  #       maskdata[maskdata==0] = roidata_weighted[maskdata==0]
        maskdata = np.maximum(maskdata, roidata_weighted)
        print(e)
      except KeyboardInterrupt:
        raise KeyboardInterrupt
      except:
        print("Failed")
        pass
    newmask = nib.Nifti1Image(maskdata, mask.affine)
    nib.save(newmask, os.path.join(WORKSPACEROOT, savename))

  clusterlabels = kmeans_model.labels_.copy()
  clusterlabels[clusterlabels == 0] = np.max(clusterlabels)+1
  genmasks(clusterlabels, f"P06cort_cluster_total.nii.gz")
  [genmasks(clusterlabels == i, f"P06cort_cluster_{i}.nii.gz") for i in range(1,np.max(clusterlabels)+1)]



if __name__ == "__main__":
  part05()