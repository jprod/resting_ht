#!/usr/bin/env python

"""
Clusters correlation matrix and produces clusters masks based on atlas rois for cortical rois 
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

def part06(TOTALMASKROOT = r"F:\+DATA\ABSFULL\TOTAL ATLAS", k_clust=None, resample_space=r"F:\+DATA\DOCKEROUT\MNI152_T1_1mm_brain.nii.gz", msk_prefix = "HTcomm_rerun"):
  """
  Clusters correlation matrix and produces clusters masks based on atlas rois for cortical rois.

  Parameters
  ----------
  TOTALMASKROOT : str
    Directory of the atlas
  k_clust : int
    Number of clusters for k means
  resample_space : str
    Full path for a nifti in the desired resample space
  msk_prefix : str
    Prefix for current generated HT masks as part of cortical roi atlas

  Returns
  -------
  None (Saves output to "results" directory)
  Output includes 
    P06_clusterlabels.npy - roi array for cluster assignments
    P06cort_cluster_total.nii.gz - 4d nifti of combined mask of rois per each clusters asignment
    P06cort_cluster_{i}.nii.gz - 3d nifti mask of combined rois for each cluster

  Notes
  -----
  Default arguments are currently set for HT analysis on author's personal computer.

  """
  print("--Running part 06--")

  resample_threshold = 0.2

  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')


  meancorr_in = np.load(os.path.join(WORKSPACEROOT, 'P05_fishercrosssubtotalTCcorr.npy'))
  corr_in = np.load(os.path.join(WORKSPACEROOT, 'P05_fisherTCcorr.npy'))
  print(f"meancorr_in shape: {meancorr_in.shape}")
  ROIORDER = np.load(os.path.join(WORKSPACEROOT, 'P05_ROIorder.npy'))
  print(f"ROIORDER shape: {ROIORDER.shape}")

  # regex for cortical
  import re
  r = re.compile('[lr]h\.[RL]_.*')
  cortex = np.vectorize(lambda name: bool(re.match(r, name)))(ROIORDER)
  # TODO: Edit so it uses non HT indicies

  r2 = re.compile(f'.*{msk_prefix}.*')
  TargetRoiIndices = np.vectorize(lambda name: bool(re.match(r2, name)))(ROIORDER)
  print(TargetRoiIndices)
  print(f'.*{msk_prefix}.*')
  # TargetRoiIndices = np.where(np.isin(ROIORDER, ["HTcomm_1", "HTcomm_2", "HTcomm_3", "HTcomm_4"]))[0]

  print(cortex)
  print(cortex.shape)

  meancorr = meancorr_in[TargetRoiIndices,:]
  meancorr = meancorr[:,cortex]
  corr = corr_in[:,TargetRoiIndices,:]
  corr = corr[:,:,cortex]
  ROIORDER2 = ROIORDER[cortex]

  corr_moved = np.moveaxis(corr, -1, 0)
  meancorr_moved = np.moveaxis(meancorr, 0, -1)
  print(f"corr_moved shape: {corr_moved.shape}")
  print(f"corr shape: {corr.shape}")
  print(f"ROIORDER2: {ROIORDER2}")
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
    
  if k_clust is None:
    k_clust = kneedle.knee

  kmeans_model = KMeans(n_clusters=k_clust, random_state=1).fit(meancorr_moved)
  print(kmeans_model.labels_)
  for i in range(max(kmeans_model.labels_)+1):
    print(f"{i}: {sum(kmeans_model.labels_ == i)}")
  for i in range(kneedle.knee):
    print(f"{i}")
    print(ROIORDER2[kmeans_model.labels_ == i])

  def avg_cluster(clusterset_x_numroi):
    cluster_n = np.max(clusterset_x_numroi)+1
    avgcluster = np.stack([np.nanmean(corr_moved[clusterset_x_numroi == i], axis=0) for i in range(cluster_n)])
    return(avgcluster)
  corrcluster = avg_cluster(kmeans_model.labels_)

  from statsmodels.stats.anova import AnovaRM
  def getdfanovacluster(clustxsubxnewroi_matrix, savepath="out.csv"):
    nsub = clustxsubxnewroi_matrix.shape[1]
    nclust = clustxsubxnewroi_matrix.shape[0]
    subxnewroi_matrix = np.concatenate([subxnewroimat for subxnewroimat in clustxsubxnewroi_matrix], axis=1)
    subid = np.arange(nsub).reshape((-1,1))
    full = np.concatenate([subid, subxnewroi_matrix], axis=1)
    print(full)
    print(full.shape)
    long = np.concatenate([np.concatenate([full[:,[0,i]], np.full((nsub,1), (i-1)%nclust+1),
                                           np.full((nsub,1), (i-1)//nclust+1)], axis=1) for i in range(1, full.shape[1])], axis=0)
    print(long)
    print(long.shape)
    dflong = pd.DataFrame(long, columns=["pid", "rvalue", "subregion", "cluster"])
  #   dflong.to_csv(savepath) 
    dffull = pd.DataFrame(full, columns=["pid"]+[f"rvalue_clust{i//nclust + 1}_subreg{i%nclust + 1}" for i in range(1, full.shape[1])])
    dffull.to_csv(savepath, index=False)
    aovrm = AnovaRM(dflong, depvar='rvalue', subject='pid', within=['subregion', 'cluster'])
    res = aovrm.fit()
    print(res)
    return(np.stack([res.anova_table["F Value"].to_numpy(), res.anova_table["Pr > F"].to_numpy()], axis=0))

  anovares2 = getdfanovacluster(corrcluster, os.path.join(WORKSPACEROOT, "P06cort_clustersWIDE.csv"))

  os.chdir(TOTALMASKROOT)
  maskpaths = [file for file in glob.glob("**/*.nii.gz", recursive=True)]
  # print(maskpaths)
  def genmasks(row, savename, maskorder=ROIORDER2):
    """Must have WORKSPACEROOT, TOTALMASKROOT and ROILIST defined."""
    os.chdir(TOTALMASKROOT)
    print(row)
    mask = nib.load(resample_space)
    # maskdata = np.zeros((176, 208, 176))
    maskdata = np.zeros(mask.dataobj.shape)
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
        print(" - loaded", end="")
        if looproi[2] == '.' or looproi[0] == 'W' or looproi[1:3] == 'HI': 
          roiresampled = resample_img(roimask, target_shape=mask.shape[:3], target_affine=mask.affine, 
            interpolation='nearest') # use linear neighbor interpolation for probabilistic images
          print(" - resampled")
          roiresampled = math_img(f'img > 0.2', img=roiresampled)
  #         view = plotting.view_img(roiresampled, threshold=0)
          roidata = roiresampled.get_fdata()
        elif looproi[1] == '_':      
          roiresampled = resample_img(roimask, target_shape=mask.shape[:3], target_affine=mask.affine, 
            interpolation='linear') # use linear neighbor interpolation for probabilistic images
          print(" - resampled")
          roiresampled = math_img(f'img > 0.2', img=roiresampled)
  #         view = plotting.view_img(roiresampled, threshold=0)
          roidata = roiresampled.get_fdata()
        else:
  #         print(roimask)
          print("")
          roidata = roimask.get_fdata()
        print("extracted roidata")
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
  clusterlabels = clusterlabels + 1
  print(clusterlabels)
  with open(os.path.join(WORKSPACEROOT, "P06_clusterlabels.npy"), 'wb') as f:
    np.save(f, clusterlabels)
  print("P06: Saved Labels")
  genmasks(clusterlabels, f"P06cort_cluster_total.nii.gz")
  [genmasks(clusterlabels == i, f"P06cort_cluster_{i}.nii.gz") for i in range(1,np.max(clusterlabels)+1)]

if __name__ == "__main__":
  part06()