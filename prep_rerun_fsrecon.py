#!/usr/bin/env python3
"""
Construct full edited FS recons for each subject/editor and rerun with appropriate recon-all flags

Author : Mike Tyszka
Place : Caltech
Dates : 2020-05-04 JMT From scratch
        2020-05-25 JMT Add insertion of edited data
"""

import os
import numpy as np
import pandas as pd
from nibabel.freesurfer import io
import shutil
from glob import glob

def main():

    # Scan assignments directory for editors and subjects
    der_dir = '/data2/conte/derivatives'
    fs_dir = os.path.join(der_dir, 'freesurfer_6')
    fs_edit_dir = os.path.join(der_dir, 'freesurfer_6_edited')
    edit_dir = os.path.join(fs_edit_dir, '+Training+')

    print('Derivatives directory : {}'.format(der_dir))
    print('Original FS subjects  : {}'.format(fs_dir))
    print('Edited FS subjects    : {}'.format(fs_edit_dir))
    print('Editor results        : {}'.format(edit_dir))

    # Init command list
    cmd_list = []

    for editor in os.listdir(edit_dir):

        print('')
        print('{}'.format(editor))

        for subject in os.listdir(os.path.join(edit_dir, editor)):

            print('  {}'.format(subject))

            # Find unedited FS recon in main repository
            subj_dir = os.path.join(fs_dir, subject)

            if not os.path.isdir(subj_dir):

                print('* {} not found amongst unedited subjects - skipping'.format(subject))

            else:

                # Create a new per-editor clone of the original unedited FS recon
                edited_subject = '{}-{}'.format(subject, editor)
                subj_edit_dir = os.path.join(fs_edit_dir, edited_subject)

                # Check whether a clone already exists
                if os.path.isdir(subj_edit_dir):

                    print('  {} already exists - skip cloning'.format(subj_edit_dir))

                else:

                    # Clone unedited recon to edited directory
                    print('  Cloning {} to {}'.format(subject, fs_edit_dir))
                    shutil.copytree(subj_dir, subj_edit_dir)

            # Init recon-all command for rerun
            fs_cmd = 'recon-all -sd {} -subjid {}'.format(fs_edit_dir, edited_subject)
            arpial_opt = ''
            ar3_opt = ''

            # Find edited data for this editor and subject
            src_brain_mask = os.path.join(edit_dir, editor, subject, 'brainmask.mgz')
            if os.path.isfile(src_brain_mask):
                dst_brain_mask = os.path.join(subj_edit_dir, 'mri', 'brainmask.mgz')
                print('  Copying brain mask')
                print('    From : {}'.format(src_brain_mask)) 
                print('    To   : {}'.format(dst_brain_mask)) 
                shutil.copyfile(src_brain_mask, dst_brain_mask)
                arpial_opt = ' -autorecon-pial'

            src_brain_man = os.path.join(edit_dir, editor, subject, 'brain.finalsurf.manedit.mgz')
            if os.path.isfile(src_brain_man):
                dst_brain_man = os.path.join(subj_edit_dir, 'mri', 'brain.finalsurf.manedit.mgz')
                print('  Copying brain manual edit')
                print('    From : {}'.format(src_brain_man)) 
                print('    To   : {}'.format(dst_brain_man)) 
                shutil.copyfile(src_brain_man, dst_brain_man)
                arpial_opt = ' -autorecon-pial'

            src_wm_mask = os.path.join(edit_dir, editor, subject, 'wm.mgz')
            if os.path.isfile(src_wm_mask):
                dst_wm_mask = os.path.join(subj_edit_dir, 'mri', 'wm.mgz')
                print('  Copying white matter mask')
                print('    From : {}'.format(src_wm_mask)) 
                print('    To   : {}'.format(dst_wm_mask)) 
                shutil.copyfile(src_wm_mask, dst_wm_mask)
                fs_cmd += ' -autorecon2-wm'
                ar3_opt = ' -autorecon3'

            src_wm_cps = os.path.join(edit_dir, editor, subject, 'control.dat')
            if os.path.isfile(src_wm_cps):
                # Safely create tmp directory for control points
                tmp_dir = os.path.join(subj_edit_dir, 'tmp')
                os.makedirs(tmp_dir, exist_ok=True)
                dst_wm_cps = os.path.join(tmp_dir, 'control.dat')
                print('  Copying brain mask')
                print('    From : {}'.format(src_wm_cps)) 
                print('    To   : {}'.format(dst_wm_cps)) 
                shutil.copyfile(src_wm_cps, dst_wm_cps)
                fs_cmd += ' -autorecon2-cp'
                ar3_opt = ' -autorecon3'

            # Complete options
            fs_cmd += ar3_opt + arpial_opt

            # Add freesurfer command to job list
            cmd_list.append(fs_cmd)

    # Write command list
    cmds_fname = 'rerun_fsrecon.cmds' 
    print('Writing Freesurfer commands to {}'.format(cmds_fname))
    with open(cmds_fname, 'w') as f:
        f.write('\n'.join(cmd_list))

    print('Done')
    

if '__main__' in __name__:

    main()
