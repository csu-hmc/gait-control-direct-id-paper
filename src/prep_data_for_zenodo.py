#!/usr/bin/env python

"""
This script packages the raw data collected during the perturbation
experiments into a single gzipped tarball which retains the directory
structure and only includes the specific files needed for public
dissemination.

To use::

    $ python prep_data_for_zenodo.py <path-to-data-dir>

Zenodo has a default file size limit of 2GB, so I may need to split this up
into multiple tar balls each under 2GB.

Figshare has a filesize limit of 250mb.
"""

import os
import sys
import re
import tarfile
from collections import OrderedDict

# T001 and T002 where exploratory data provided by Obinna and Ton and
# were not collected with the protocol for the perturbation experiments,
# so we exclude them.
trials_not_to_include = ['T001', 'T002']

# Only include files that fit thiss naming convention.
file_regexes = ['README.rst',
                'meta-\d{3}.yml',
                'mocap-\d{3}.txt',
                'record-\d{3}.txt',
                #'cortex-\d{3}.zip',
                #'cortex-\d{3}-\d{2}.zip',
                'TestingProtocol-\d{3}.caren',
                'TestingProtocol-\d{3}.dflow',
                'TestingProtocol-\d{3}.sceneconfig']

file_regexes = [re.compile(regex) for regex in file_regexes]


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            matches = [True if regex.match(f) else False
                       for regex in file_regexes]
            if any(matches):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size


def group_trial_dirs_by_size(directory):
    exp = re.compile('T\d{3}')

    trial_dirs = sorted([d for d in os.listdir(directory)
                         if exp.match(d) and d not in trials_not_to_include])

    sizes = [get_directory_size(os.path.join(directory, trial_dir))
             for trial_dir in trial_dirs]

    size_of_dirs = 0
    start_idx = 0
    groups = OrderedDict()
    file_idx = 1

    for i, dir_size in enumerate(sizes):
        size_of_dirs += dir_size
        # put them into ~1GB chunks (should be compressed ~ 4X)
        if size_of_dirs > 1e9:
            groups[file_idx] = trial_dirs[start_idx:i]
            # Reset
            start_idx = i
            size_of_dirs = 0
            file_idx += 1

    return groups


def build_tar_ball(directory):

    groups = group_trial_dirs_by_size(directory)

    base_tar_file_name = 'perturbed-walking-data-{:0>2}.tar.gz'

    for group_num, trial_dirs in groups.items():

        tar_file_name = base_tar_file_name.format(group_num)

        with tarfile.open(tar_file_name, 'w:gz') as tar:
            print('Creating {}'.format(tar_file_name))

            for dirpath, dirnames, filenames in os.walk(directory):

                for filename in filenames:

                    matches = [True if regex.match(filename) else False for
                               regex in file_regexes]

                    subdir = os.path.basename(os.path.normpath(dirpath))

                    if any(matches) and subdir in trial_dirs:
                        path_to_file = os.path.join(dirpath, filename)
                        tar.add(path_to_file, arcname=os.path.join(subdir,
                                                                   filename))
                        print('Added {} to {}'.format(os.path.join(dirpath,
                                                                   filename),
                                                      tar_file_name))

if __name__ == "__main__":

    directory = sys.argv[1]
    print('Creating tar ball of: {}'.format(os.path.abspath(directory)))
    build_tar_ball(directory)
