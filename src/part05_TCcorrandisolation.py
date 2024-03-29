#!/usr/bin/env python

"""
Correlates fisher transformed timecourses for roi data
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

def part05(timecourse_root = r"F:\+DATA\ABSFULL\timecourse_mk4", msk_prefix = r"rest_HTABS"):
  """
  Correlates fisher transformed timecourses for roi data.

  Parameters
  ----------
  timecourse_root : str
    Housing directory for timecourse files
  msk_prefix : str
    Prefix for hypothalamus mask

  Returns
  -------
  None (Saves output to "results" directory)
  Output includes 
    P05_fisherTCcorr.npy - sub x roi x roi tensor for correlations between timecourses
    P05_fishercrosssubtotalTCcorr.npy - roi x roi matrix meaned over sub

  Notes
  -----
  Default arguments are currently set for HT analysis on author's personal computer.

  """
  print("--Running part 05--")
  # Getting File Paths  -----------------------------------------------------------------
  # OS Paths
  cwd = Path(os.getcwd())
  parent = cwd.parent.absolute()
  WORKSPACEROOT = os.path.join(parent, 'results')

  os.chdir(timecourse_root)
  tcpaths = [file for file in glob.glob("**/*res4d.csv", recursive=True)]
  # print(tcpaths)

  header = ["subj", "tag", "roi"] + [f"{i}_mean" for i in range(0,256)]

  maxlen_exp = 609

  # Extraction and Cleaning
  # Basic extraction fns for tc data --------------------
  def read_tc(p, disppaths=False):
    if disppaths: 
      print(p)
    if p[30:35] == 'HTABS':
      return pd.read_csv(p, usecols=header).drop(["subj"], axis=1)
    elif p[25:32] == 'columns' or p[25:34] == "rosmedcau":
      return pd.read_csv(p).drop(["subj"], axis=1).rename(columns=lambda x: x + "_mean" if x != "roi" else x)
    else:
      return pd.read_csv(p).drop(["subj","cat_N","median_x","median_y","median_z","roi_sd"], axis=1).sort_values("roi")

  def getRunTable(sub, runnum, drop_metacols=True, residuals=False, disppaths=False):
    subpaths = [file for file in glob.glob(f"rest0{runnum}*/**/*{sub}*", recursive=True)]
    if len(subpaths) < 2:
      return None
    csvs = [read_tc(p, disppaths) for p in subpaths]
    fulltable = pd.concat(csvs,ignore_index=True)
    if residuals:
      ftpd = fulltable.drop(["tag","roi"], axis=1).dropna(axis='columns')
      ft = ftpd.to_numpy()
      GS = np.mean(ft,0)
      def yhat_perrow(row):
        reg = LinearRegression().fit(GS.reshape(-1, 1), row.reshape(-1, 1))
        yhat = reg.predict(GS.reshape(-1, 1))
  #       sns.scatterplot(x=GS.reshape(-1,), y=row.reshape(-1,))
  #       plt.plot(GS, yhat, color='r')
  #       plt.show()
        return yhat
      ftyhat = np.apply_along_axis(yhat_perrow, -1, ft).reshape(ft.shape)
      return pd.DataFrame(data=ft-ftyhat,columns=ftpd.columns)
    elif drop_metacols:
      return fulltable.drop(["tag","roi"], axis=1)
    else:
      return fulltable
    
  # def getMetaCols(sub): [getMetaCols(sub), getRunTable(sub, 1), getRunTable(sub, 2), getRunTable(sub, 3)]
  #   listdf = [getRunTable(sub, 1, False),getRunTable(sub, 2, False),getRunTable(sub, 3, False)]
  #   print(".",end="")
  #   fulltab = pd.concat([df[["tag","roi"]] for df in listdf if df is not None], axis=1)
  #   if len(fulltab["roi"].shape) == 2:
  #     valid_check = fulltab["roi"].isin(fulltab["roi"].iloc[:, 1]).all(1).all()
  #     if not valid_check:
  #        display(fulltab)
  #        raise ValueError(f"{sub}: Cols must have the same order")
  #   return fulltab.iloc[:, 0:2]

  def getFullTable(sub):
    listdf = [getRunTable(sub, 1, False),getRunTable(sub, 2, False),getRunTable(sub, 3, False)]
    print(".",end="")
    fulltab = pd.concat([df[["tag","roi"]] for df in listdf if df is not None], axis=1)
    if len(fulltab["roi"].shape) == 2:
      valid_check = fulltab["roi"].isin(fulltab["roi"].iloc[:, 1]).all(1).all()
      if not valid_check:
        print(fulltab)
        maxlen = max([df.shape[0] for df in listdf if df is not None])
        if maxlen != maxlen_exp:
          raise ValueError(f"{sub}: maxlen {maxlen} is not {maxlen_exp}")
        listdf = [df for df in listdf if df.shape[0] == maxlen]
        fulltab = pd.concat([df[["tag","roi"]] for df in listdf if df is not None], axis=1)
        # raise ValueError(f"{sub}: Cols must have the same order")
    elif (fulltab["roi"].shape[0] != maxlen_exp):
      return None
    return pd.concat([fulltab.iloc[:, 0:2]] + [df.drop(["tag","roi"], axis=1) for df in listdf if df is not None], axis=1, join="inner")

  subjset = set([p[-17:-10] for p in tcpaths]) # Where in the path the subject is stated
  print("P05: Loading All Runs...")
  allruns = [(sub, getFullTable(sub)) for sub in subjset]
  allruns = [(sub, run) for sub, run in allruns if run is not None]
  print("P05: Loaded All Runs")
  ROILIST = allruns[0][1]['roi']
  ROINP = ROILIST.to_numpy().astype('U')
  with open(os.path.join(WORKSPACEROOT, 'P05_ROIorder.npy'), "wb") as f:
    np.save(f, ROINP)

  # Correlation
  print("P05: Correlation")
  # TODO: only works for set mask prefixes as a heuristic check for index order
  allcorr = [(subj[1].drop(["tag", "roi"], axis=1).transpose().corr(), subj[1][subj[1]["tag"] == msk_prefix].index, subj[0]) for subj in allruns] 
  corrvalid = [corr[0] for corr in allcorr if (corr[1] == allcorr[0][1]).all()] # valid if the indices are the same
  # Checking if any correlation tables are malformed
  for corr in allcorr:
   if len(corr[1]) == 0:
      print(corr[0])
      print(corr[2])
  corrnp = np.stack([corr.to_numpy() for corr in corrvalid])
  fd = lambda mat: np.fill_diagonal(mat,0)
  corrnpcopy = corrnp.copy()
  [fd(mat) for mat in corrnpcopy]
  fishercorrnp = np.arctanh(corrnpcopy)
  with open(os.path.join(WORKSPACEROOT, "P05_fisherTCcorr.npy"), "wb") as f:
      np.save(f, fishercorrnp)

  # Mean
  print("P05: Mean")
  meancorr = np.nanmean(fishercorrnp, axis=0)
  with open(os.path.join(WORKSPACEROOT, "P05_fishercrosssubtotalTCcorr.npy"), 'wb') as f:
    np.save(f, meancorr)





if __name__ == "__main__":
  part05()