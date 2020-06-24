#!/usr/bin/env python3
"""
Compute intersurface distances between two Freesurfer surfaces
-

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
from datetime import datetime as dt

from nibabel.freesurfer.io import (read_geometry, read_morph_data, write_morph_data)
from sklearn.metrics import pairwise_distances_argmin_min

# from scipy.spatial.distance import (directed_hausdorff, euclidean)


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute intersurface distance statistics with annotation map')
    parser.add_argument('-sd', '--subjdir', required=False, help='FS subjects directory')
    parser.add_argument('-s1', '--subj1', required=True, help='First subject ID')
    parser.add_argument('-s2', '--subj2', required=True, help='Second subject ID')
    parser.add_argument('-hs', '--hemi', required=False, default='lh', help='Hemisphere [''lh'']')
    parser.add_argument('-sn', '--surfname', required=False, default='pial', help='Surface name [''pial'']')
    parser.add_argument('-o', '--outdir', required=True, help='Output directory')

    # Parse command line arguments
    args = parser.parse_args()

    if args.subjdir:
        subjects_dir = args.subjdir
    else:
        subjects_dir = os.getenv('SUBJECTS_DIR')

    # Construct surface paths
    surf1_fname = os.path.join(subjects_dir, args.subj1, 'surf', '{}.{}'.format(args.hemi, args.surfname))
    surf2_fname = os.path.join(subjects_dir, args.subj2, 'surf', '{}.{}'.format(args.hemi, args.surfname))

    if not os.path.isfile(surf1_fname):
        print('* Subject 1 surface file {} does not exist - exiting'.format(surf1_fname))
        sys.exit(1)

    if not os.path.isfile(surf2_fname):
        print('* Subject 2 surface file {} does not exist - exiting'.format(surf2_fname))
        sys.exit(1)

    # Load surfaces
    try:
        coords1, faces1 = read_geometry(surf1_fname)
    except Exception:
        print('* Problem loading surface from {}'.format(surf1_fname))
        sys.exit(1)

    try:
        coords2, faces2 = read_geometry(surf2_fname)
    except Exception:
        print('* Problem loading surface from {}'.format(surf2_fname))
        sys.exit(1)

    # Create output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    print('Subject 1 mesh has {} points'.format(coords1.shape[0]))
    print('Subject 2 mesh has {} points'.format(coords2.shape[0]))

    # Pairwise Euclidean distances between nodes of surface 1 and 2
    # If coords1 is N x 3 and coords2 is M x 3, Y is N x M
    print('Starting pairwise distances')
    t0 = dt.now()
    _, distmin = pairwise_distances_argmin_min(coords1, coords2)
    delta = dt.now() - t0
    print('Done in {:0.3f} seconds'.format(delta.total_seconds()))

    # Save closest distances as a morphometry/curv file
    dist_fname = os.path.join(args.outdir, '{}-{}-{}.dist'.format(args.subj1, args.subj2, args.hemi))
    print('Saving intersurface distances to {}'.format(dist_fname))
    write_morph_data(dist_fname, distmin)

    # Copy subject 1 surface to output directory for use with distance annotation in Freeview
    surf1_bname = os.path.basename(surf1_fname)
    print('Copying {} to {}'.format(surf1_bname, args.outdir))
    shutil.copy(surf1_fname, os.path.join(args.outdir, surf1_bname))

    # Fast Hausdorff distances between nodes of surface 1 and 2
    # print('Starting forward Hausdorff')
    # t0 = dt.now()
    # d12, _, _ = directed_hausdorff(coords1, coords2)
    # delta = dt.now() - t0
    # print('Done in {:0.3f} seconds'.format(delta.total_seconds()))
    #
    # print('Starting reverse Hausdorff')
    # t0 = dt.now()
    # d21, _, _ = directed_hausdorff(coords2, coords1)
    # delta = dt.now() - t0
    # print('Done in {:0.3f} seconds'.format(delta.total_seconds()))
    #
    # print('Forward Hausdorff Distance   : {:03f} mm'.format(d12))
    # print('Reverse Hausdorff Distance   : {:03f} mm'.format(d21))
    # print('Symmetric Hausdorff Distance : {:03f} mm'.format(max(d12, d21)))


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
