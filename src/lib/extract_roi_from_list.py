def extract_timecourse(subj, gm_file, func_file, out_dir, roi_path, out_label, check_output=None, dilate_roi=None, gm_method='scale', gm_thresh=None, export_nii=None):
    '''
    Extract ROI timecourse from functional data, given a list of nifti files.

    Works with 3d or 4d files, so can be applied to a timecouse, or to modeling output.

    Does not remove confounds. For that use, nilearn.img.clean_img

    [Required]
    subj = string, subject identifier, e.g. sub-001.
        Can also use some other tag here, e.g. 'stress_lvl2_speech-prep'
    gm_file = string, full path to gm mask, in same space as functional data.
        Be sure to include hemisphere information in the name.
        e.g. 'PATH/sub-001_fmriprep_skullstrip_ref_img.nii.gz__lh_REL.nii'
    func_file = string, full path to preprocessed functional data.
        e.g. PATH/'sub-001_task-rest_run-01_bold.nii.gz'
    out_dir = string, full path to folder to save outputs.
        e.g. '/home/project/outputs'
    roi_path = string, glob path to grab all ROI files.
        e.g. os.path.join(roi_dir, '*dil_ribbon_EPI_bin_ribbon.nii.gz')
    out_label = string, to be added to output files to specify anything you want,
        e.g. wm_Glasser

    [Optional]
    check_output [default = None] = set to True to print GM masked functional data and dilated ROIs.
    dilate_roi [default = None] = set to an Integer to dilate each ROI by X voxels.
    gm_method [default = 'scale'] Enter 'scale', 'above', or 'below'.
            'scale' gives a weighted average by multiplying the functional data by the GM mask.
            'between' incldues all voxels ABOVE the first and BELOW/EQUAL to the second value (given as a list in gm_thresh)
    gm_thresh [default = None] provide a float or list to use with thresholding in gm_method
        (e.g. [.2, 1.] to grab all gm voxels > .2)
    export_nii [default = None] = set to True to output a .nii file with ROI averages.
    '''
    import os, glob, time
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from nilearn.image import resample_img
    from scipy.ndimage.morphology import binary_dilation

    print('subj: ', subj)
    print('gm_file: ', gm_file)
    print('func_file:', func_file)
    print('out_dir:', out_dir)
    print('roi_path:', roi_path)
    print('out_label:', out_label)
    print('check_output:', check_output)
    print('dilate_roi:', dilate_roi)
    print('gm_method:', gm_method)
    print('gm_thresh:', gm_thresh)
    print('export_nii:', export_nii, '\n\n')

    func_img = nib.load(func_file)
    print('linear neightbor interpolation of GM mask to functional space')
    fit_gm = resample_img(nib.load(gm_file),
                           target_affine=func_img.affine,
                           target_shape=func_img.shape[0:3],
                           interpolation='linear')

    # Set up a 4d array, to house all ROIs.
    roi_shape = [i for i in nib.load(func_file).shape[0:3]]
    roi_shape.append(len(glob.glob(roi_path)))
    roi_all = np.zeros(roi_shape)

    # loop through rois, reshaping each to functional space, and dilating if necessary.
    for idx, roi in enumerate(glob.glob(roi_path)):
        print('loading:', roi)
        if len(np.unique(nib.load(roi).get_fdata())) > 2:
            print('linear interpolation of probabalistic ROI to functional space')
            fit_roi = resample_img(nib.load(roi),
                                   target_affine=nib.load(func_file).affine,
                                   target_shape=nib.load(func_file).shape[0:3],
                                   interpolation='linear')
        else:
            print('nearest neighbor interpolation of binary ROI to functional space')
            fit_roi = resample_img(nib.load(roi),
                                   target_affine=nib.load(func_file).affine,
                                   target_shape=nib.load(func_file).shape[0:3],
                                   interpolation='nearest')

        if dilate_roi:
            print('dilate ROI by', dilate_roi, 'voxels. \n WARNING: THIS WILL REMOVE ANY PROBABLISTIC MAPPING AND SWITCH TO BINARY')
            fit_roi = nib.Nifti1Image(binary_dilation(fit_roi.get_fdata(), iterations=dilate_roi).astype(fit_roi.get_fdata().dtype),
                    fit_roi.affine, fit_roi.header)

        roi_all[...,idx] = fit_roi.get_fdata()

        if idx == 0:
            roi_median_x = np.median(np.where(roi_all[...,idx]>0)[0])
            roi_median_y = np.median(np.where(roi_all[...,idx]>0)[1])
            roi_median_z = np.median(np.where(roi_all[...,idx]>0)[2])
        else:
            roi_median_x = np.hstack((roi_median_x, np.median(np.where(roi_all[...,idx]>0)[0])))
            roi_median_y = np.hstack((roi_median_y, np.median(np.where(roi_all[...,idx]>0)[1])))
            roi_median_z = np.hstack((roi_median_z, np.median(np.where(roi_all[...,idx]>0)[2])))

    roi_all[roi_all==0] = np.nan

    if check_output: # save ROI, in case you want to check output.
        nib.save(nib.Nifti1Image(roi_all, fit_roi.affine, fit_roi.header),
                 os.path.join(out_dir, out_label+'_'+subj+'_all_roi.nii.gz'))

    # adjust length of slice loop, depending on whether image is 3d/4d
    if len(func_img.shape) > 3: # Use the 4th dimension, if possible.
        TR_len = func_img.shape[3]
    else:
        TR_len = 1
        func_dat = func_img.get_fdata()
        func_dat = func_dat[...,None]

    for TR in range(0, TR_len):
        print('working on functional slice:', TR)
        if len(func_img.shape)==3: # trigger on 3d images.
            func_dat = func_img.get_fdata()
        else:
            func_dat = func_img.dataobj[..., TR] # grab one slice at a time.

        print('apply GM mask to functional data.')
        if gm_method == 'scale':
            func_dat = func_dat*fit_gm.get_fdata()
        elif gm_method == 'between':
            assert isinstance(gm_thresh, list), 'for gm_method=between, use a list of two float cutoff points for gm_thresh'
            func_dat = func_dat*np.where((fit_gm.get_fdata() > gm_thresh[0]) & (fit_gm.get_fdata() <= gm_thresh[1]), 1, np.nan)
        else:
            raise('Use either scale or between for gm_method')

        print('extract all rois')
        TR_dat = roi_all*func_dat[...,None] # multiplied to handle probabalistic masks.
        TR_dat = TR_dat.reshape(np.prod(TR_dat.shape[0:-1]), -1)

        print('saving TR average')
        if TR == 0:
            TR_mean = np.nanmean(TR_dat, axis=0)
        else:
            TR_mean = np.vstack((TR_mean, np.nanmean(TR_dat, axis=0))) # ROI x TRs

    # export data to .csv
    pd_out = pd.DataFrame({'subj':np.repeat(subj, len(glob.glob(roi_path))),
                           'tag':np.repeat(out_label, len(glob.glob(roi_path))),
                            'roi':[f.split('/')[-1] for f in glob.glob(roi_path)],
                            'cat_N':np.sum(~np.isnan(TR_dat), axis=0),
                            'median_x':roi_median_x,
                            'median_y':roi_median_y,
                            'median_z':roi_median_z})
    if TR_len > 1: # no STD on 3d images.
        pd_out['roi_sd'] = np.nanstd(TR_mean, axis=0)
        pd_out = pd_out.join(pd.DataFrame(TR_mean.transpose()).add_suffix('_mean'))
    else:
        pd_out = pd_out.join(pd.DataFrame(TR_mean).add_suffix('_mean'))
    pd_out.to_csv(os.path.join(out_dir,
                               os.path.join(out_dir, out_label+'_'+subj+'_'+func_file.split('/')[-1].split('.nii.gz')[0]+'.csv')),
                  index=False, header=True)

    if export_nii: # export nifti
        nii_mean = np.zeros(nib.load(func_file).shape[0:3])
        print('writing nifti for ROI means.')
        for idx, roi in enumerate(glob.glob(roi_path)):
            if TR_len > 1:
                nii_mean[roi_all[...,idx]>0] = np.mean(TR_mean[:,idx]) # This is mean of voxel means across TRs.
            else:
                nii_mean[roi_all[...,idx]>0] = np.mean(TR_mean[idx]) # This is mean of voxel means across TRs.
        nii_mean_nib = nib.Nifti1Image(nii_mean, nib.load(func_file).affine, nib.load(func_file).header)
        nii_mean_nib.header['cal_max'] = np.nanmax(nii_mean) # adjust min and max header info.
        nii_mean_nib.header['cal_min'] = np.nanmin(nii_mean)
        nib.save(nii_mean_nib,
                 os.path.join(out_dir, out_label+'_'+subj+'_mean_'+func_file.split('/')[-1]))
        if TR_len > 1:
            nii_sd = np.zeros(nib.load(func_file).shape[0:3])
            nii_sd[roi_all[...,idx]>0] = np.std(TR_mean[:,idx]) # This is mean of voxel means across TRs.
            nii_sd_nib = nib.Nifti1Image(nii_sd, nib.load(func_file).affine, nib.load(func_file).header)
            nii_sd_nib.header['cal_max'] = np.nanmax(nii_sd) # adjust min and max header info.
            nii_sd_nib.header['cal_min'] = np.nanmin(nii_sd)
            nib.save(nii_sd_nib,
                     os.path.join(out_dir, out_label+'_'+subj+'_sd_'+func_file.split('/')[-1]))

    print('####\ndone with %s \n####' % subj)

# import glob, os
# subj = os.environ['SUBJ']
# root = '/scratch/'+os.environ['USER']+'/'+os.environ['SUBJ']+'/'+os.environ['PROJNAME']
# gm_file = glob.glob(os.path.join(root, 'gm/*.nii.gz'))[0]
# func_file = os.path.join(root, 'func/res4d.nii.gz')
# # roi_path = os.path.join(root, 'roi/*')
# out_dir = os.path.join(root, 'output')
# # out_label = 'wm_GlasserPauli'

# roi_groups = [
#     # ['rest_Glasser_L', os.path.join(root, 'roi1/*')],
#     # ['rest_Glasser_R', os.path.join(root, 'roi2/*')],
#     # ['rest_PauliPAG', os.path.join(root, 'roi3/*')],
#     # ['rest_WAGERbrainstem', os.path.join(root, 'roi4/*')],
#     # ['rest_WAGERcerebellum', os.path.join(root, 'roi5/*')],
#     # ['rest_WAGERsubcortex', os.path.join(root, 'roi6/*')]
#     # ['rest_Yuta_hippocampus', os.path.join(root, 'roi7/*')]
#     ['rest_Ajay_AP_hippocampus', os.path.join(root, 'roi8/*')]
#     ]
# subj = subj[0:7]
# for roi_g in roi_groups:
#     out_label = roi_g[0]
#     roi_path = roi_g[1]
#     extract_timecourse(subj, gm_file, func_file, out_dir,
#                           roi_path, out_label,
#                           gm_method='between', gm_thresh=[.2, 1.])
