#!/usr/bin/env python3
"""
Merge two sets of Freeview/Freesurfer control points
- used during editing for adjusting bias correction of white matter segment
- loads both control point text files, identifies unique points in union and writes to text file

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
import numpy as np


def load_cps(fname):
    """
    Load contents of FS7 control point file

    Example control.dat file:
    58.4497 -6.64394 -14.5253
    60.4497 -7.64394 -16.5253
    58.4497 -5.64394 -13.5253
    58.4497 -2.64394 -11.5253
    58.4497 -9.64394 -16.5253
    57.4497 -7.64394 -14.5253
    60.4497 -6.64394 -16.5253
    -49.5503 -7.64394 -19.5253
    -46.5503 -5.64394 -17.5253
    info
    numpoints 9
    useRealRAS 1
    """

    cps = []

    try:
        with open(fname, 'r') as fd:
            for line in fd:

                # Split line into space-separated values
                tmp = line.strip()
                vals = tmp.split(' ')

                if 'numpoints' in vals[0]:
                    npnts = int(vals[1])

                elif 'useRealRAS' in vals[0]:
                    use_real_ras = bool(vals[1])

                elif 'info' in vals[0]:
                    pass

                elif len(vals) == 3:
                    cps.append(vals)

                else:
                    pass

    except IOError:
        print('* Problem loading {}'.format(fname))
    except UnicodeDecodeError:
        print('* Problem decoding {}'.format(fname))

    return np.array(cps, dtype=float), npnts, use_real_ras


def save_cps(fname, cps, ras_flag):

    with open(fname, 'w') as fd:
        for pnt in cps:
            fd.write(' '.join(['{:f}'.format(x) for x in pnt]))
            fd.write('\n')
        fd.write('info\n')
        fd.write('numpoints {:d}\n'.format(cps.shape[0]))
        fd.write('useRealRAS {:d}\n'.format(int(ras_flag)))


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge Freesurfer control point files')
    parser.add_argument('-i1', '--infile1', required=True, help="Freesurfer control point text file #1")
    parser.add_argument('-i2', '--infile2', required=True, help="Freesurfer control point text file #2")
    parser.add_argument('-o', '--outfile', help="Merged control point output text file ['control_merge.dat']")

    # Parse command line arguments
    args = parser.parse_args()
    cp1_fname = args.infile1
    if not os.path.isfile(cp1_fname):
        print('* {} does not exist - exiting'.format(cp1_fname))
        sys.exit(1)

    cp2_fname = args.infile2
    if not os.path.isfile(cp2_fname):
        print('* {} does not exist - exiting'.format(cp2_fname))
        sys.exit(1)

    if args.outfile:
        merge_fname = args.outfile
    else:
        merge_fname = 'control_merge.dat'

    print('Loading {}'.format(cp1_fname))
    cp1, npnts1, ras_flag1 = load_cps(cp1_fname)

    print('Loading {}'.format(cp2_fname))
    cp2, npnts2, ras_flag2 = load_cps(cp2_fname)

    if not ras_flag1 == ras_flag2:
        print('useRealRAS flags differ between files - exiting')
        sys.exit(2)

    # Minimum separation for points to be considered distinct (mm)
    d_tol = 0.01

    # Loop over points in second set, checking for distance to points in first set
    cps = cp1.copy()
    d_min = 1e30

    for p2 in cp2:

        for p1 in cp1:
            d = np.linalg.norm(p1-p2)
            if d < d_min:
                d_min = d

        if d_min > d_tol:
            cps = np.vstack([cps, p2])

    print('Merge summary')
    print('  {} points in {}'.format(cp1.shape[0], cp1_fname))
    print('  {} points in {}'.format(cp2.shape[0], cp2_fname))
    print('  {} points in {}'.format(cps.shape[0], merge_fname))

    # Save merged point set
    print('Writing merged point set to {}'.format(merge_fname))
    save_cps(merge_fname, cps, ras_flag1)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
