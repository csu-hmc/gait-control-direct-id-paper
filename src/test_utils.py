#!/usr/bin/env python

import os
import shutil
import tempfile

import numpy as np
import yaml
import pandas

from .utils import generate_meta_data_tables


def test_generate_meta_data_tables():

    d = {'a': 1,
         'b': 2,
         'c': {'d': 3,
               'e': 4,
               'f': {'g': 5,
                     'h': 6}},
         'f': {'i': 7,
               'j': 8}}

    c = {'a': 11,
         'c': {'d': 33,
               'e': 44,
               'f': {'g': 55,
                     'h': 66}},
         'f': {'i': 77,
               'j': 88}}

    tmp_dir = tempfile.mkdtemp()

    T003 = os.path.join(tmp_dir, 'T003')
    os.mkdir(T003)

    T005 = os.path.join(tmp_dir, 'T005')
    os.mkdir(T005)

    with open(os.path.join(T003, 'meta-003.yml'), 'w') as f:
        yaml.dump(d, f)
    with open(os.path.join(T005, 'meta-005.yml'), 'w') as f:
        yaml.dump(c, f)

    tables = generate_meta_data_tables(tmp_dir)

    expected = {'TOP': {'a': [1, 11],
                        'b': [2, np.nan]},
                'TOP|c': {'d': [3, 33],
                          'e': [4, 44]},
                'TOP|f': {'i': [7, 77],
                          'j': [8, 88]},
                'TOP|c|f': {'g': [5, 55],
                            'h': [6, 66]}}

    for k, v in expected.items():
        expected[k] = pandas.DataFrame(v, index=['003', '005'])

    for k, v in tables.items():
        assert v == expected[k]

    # TODO : This should be in a tear down function.
    shutil.rmtree(tmp_dir)
