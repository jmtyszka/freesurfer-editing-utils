#!/usr/bin/env python3
"""
Compute intersurface distances between two Freesurfer surfaces

Requirements:
Strict edited subject naming convention: <base subject name>-<editor name>
Examples :
    sub-CC0016_core1-MikeT
    sub-CC0016_core1-DoritK
    sub-CC0006_core2-MikeT

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

MIT License

Copyright (c) 2020 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
from datetime import datetime as dt

from nibabel.freesurfer.io import (read_geometry, write_morph_data)
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import pairwise_distances_argmin_min

import multiprocessing as mp
import psutil

from glob import glob


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute intersurface distance statistics with annotation map')
    parser.add_argument('-sd', '--subjdir', required=True, help='Freesurfer edited subjects directory')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory')

    # Parse command line arguments
    args = parser.parse_args()

    if args.subjdir:
        subjects_dir = args.subjdir
    else:
        subjects_dir = os.getenv('SUBJECTS_DIR')

    # Create output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # Parse subject directory listing for unique subjects and editors
    subjects, editors = parse_subj_dirs(subjects_dir)

    n_subjects = len(subjects)
    n_editors = len(editors)

    # Multiprocessing setup
    n_cpu = psutil.cpu_count(logical=False)
    print('Creating pool of {} processes'.format(n_cpu))
    pool = mp.Pool(n_cpu)

    # Hemisphere and surface names
    hemis = ['lh', 'rh']
    surfnames = ['pial', 'white']

    compare_args = []
    for subject in subjects:
        for hemi in hemis:
            for surfname in surfnames:

                # Upper triangle loop for editor pairs
                for ic in range(0, n_editors-1):
                    for jc in range (ic+1, n_editors):

                        editor1 = editors[ic]
                        editor2 = editors[jc]

                        compare_args.append((subjects_dir, args.outdir, subject, editor1, editor2, hemi, surfname))

    # Submit jobs
    # result : list of [subject, editor1, editor2, hemi, surfname, d12, d21, dsym] for each job
    result = pool.starmap(compare_editors, compare_args)

    # Close pool for additional jobs and wait for completion
    pool.close()
    pool.join()

    # Save results list
    results_csv = os.path.join(args.outdir, 'Hausdorff_Distances.csv')
    df = pd.DataFrame(result, columns=['Subject', 'Editor1', 'Editor2', 'Hemisphere', 'Surface', 'D12', 'D21', 'DSYM'])
    df.to_csv(results_csv, sep=',', index=False)

def parse_subj_dirs(subjects_dir):

    print('Scanning {} for subjects and editors'.format(subjects_dir))

    # Get list of sub-* subdirectories of FS subjects directory
    dir_list = glob(os.path.join(subjects_dir, 'sub-CC*'))

    # Init running subject ID and editor name lists
    subject_list = []
    editor_list = []

    # Get full list of subject IDs and editor names with duplication
    for dname in dir_list:

        base_dname = os.path.basename(dname)

        # Split directory name at last '-'
        subject, editor = base_dname.rsplit('-', 1)

        subject_list.append(subject)
        editor_list.append(editor)

    # Boil down to unique subjects and editors
    subjects = np.unique(subject_list)
    editors = np.unique(editor_list)

    return subjects, editors


def compare_editors(subjects_dir, outdir, subject, editor1, editor2, hemi, surfname):

    subj_dir1 = os.path.join(subjects_dir, '{}-{}'.format(subject, editor1))
    subj_dir2 = os.path.join(subjects_dir, '{}-{}'.format(subject, editor2))

    # Construct surface paths
    surf1_fname = os.path.join(subj_dir1, 'surf', '{}.{}'.format(hemi, surfname))
    surf2_fname = os.path.join(subj_dir2, 'surf', '{}.{}'.format(hemi, surfname))

    if not os.path.isfile(surf1_fname):
        print('* Subject 1 surface file {} does not exist - exiting'.format(surf1_fname))
        sys.exit(1)

    if not os.path.isfile(surf2_fname):
        print('* Subject 2 surface file {} does not exist - exiting'.format(surf2_fname))
        sys.exit(1)

    # Load surfaces
    try:
        coords1, faces1 = read_geometry(surf1_fname)
    except IOError:
        print('* Problem loading surface from {}'.format(surf1_fname))
        sys.exit(1)

    try:
        coords2, faces2 = read_geometry(surf2_fname)
    except IOError:
        print('* Problem loading surface from {}'.format(surf2_fname))
        sys.exit(1)

    print('{}-{}-{}-{} mesh has {} points'.format(subject, editor1, hemi, surfname, coords1.shape[0]))
    print('{}-{}-{}-{} mesh has {} points'.format(subject, editor2, hemi, surfname, coords2.shape[0]))

    # Fast pairwise Euclidean distances between nodes of surface 1 and 2
    # If coords1 is N x 3 and coords2 is M x 3, distmin is N x M
    print('Computing pairwise distances ({} to {})'.format(editor1, editor2))
    t0 = dt.now()
    _, dmin12 = pairwise_distances_argmin_min(coords1, coords2)
    delta = dt.now() - t0
    print('Done in {:0.3f} seconds'.format(delta.total_seconds()))

    # Calculate forward Hausdorff distance from pairwise distance results
    d12 = np.max(dmin12)

    # Fast Hausdorff distances between nodes of surface 1 and 2
    print('Computing Fast Hausdorff Distances')
    t0 = dt.now()
    d21, _, _ = directed_hausdorff(coords2, coords1)
    delta = dt.now() - t0
    print('Done in {:0.3f} seconds'.format(delta.total_seconds()))

    # Symmetric Hausdorff distance (max(d12, d21))
    dsym = max(d12, d21)

    print('Forward Hausdorff Distance   : {:0.3f} mm'.format(d12))
    print('Reverse Hausdorff Distance   : {:0.3f} mm'.format(d21))
    print('Symmetric Hausdorff Distance : {:0.3f} mm'.format(dsym))

    # Save closest distances as a morphometry/curv file
    dist_fname = os.path.join(outdir, '{}-{}-{}-{}-{}.dist'.format(subject, editor1, editor2, hemi, surfname))
    print('Saving intersurface distances to {}'.format(dist_fname))
    write_morph_data(dist_fname, dmin12)

    # Copy subject 1 surface to output directory for use with distance annotation in Freeview
    surf1_bname = os.path.basename(surf1_fname)
    surf1_outname = '{}-{}-{}'.format(subject, editor1, surf1_bname)
    print('Copying {} to {}'.format(surf1_bname, surf1_outname))
    shutil.copy(surf1_fname, surf1_outname)

    return subject, editor1, editor2, hemi, surfname, d12, d21, dsym

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
