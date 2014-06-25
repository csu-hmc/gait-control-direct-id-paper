#!/usr/bin/env python

# standard library
import os
import zipfile

# external
from scipy.io import savemat

# local
import utils
from grf_landmark_settings import settings

data_dir = '../data/nick'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for trial_number, params in settings.items():

    print('Creating data for trial {}'.format(trial_number))

    params = list(params)
    params = tuple(params + [100, True])  # 100 samples per cycle and force recomputation

    steps, other = utils.merge_unperturbed_gait_cycles(trial_number, params)

    num_cycles = steps.shape[0]

    mass = float(other['First Normal Walking']['meta_data']['subject']['mass'])
    speed = other['First Normal Walking']['meta_data']['trial']['nominal-speed']

    mean_of_cycles = steps.mean(axis='items')
    std_of_cycles = steps.std(axis='items')

    percent_gait_cycle = mean_of_cycles.index.values.astype(float)

    data_dict = {'percent_gait_cycle': percent_gait_cycle,
                 'mass': mass,
                 'speed': speed,
                 'num_cycles': num_cycles}

    cols = ['Right.Ankle.PlantarFlexion.Angle',
            'Right.Ankle.PlantarFlexion.Rate',
            'Right.Ankle.PlantarFlexion.Moment',
            'RightBeltSpeed']

    for col in cols:
        data_dict[col.replace('.', '_') + '_Mean'] = mean_of_cycles[col].values
        data_dict[col.replace('.', '_') + '_Std'] = std_of_cycles[col].values

    filename = 'T{}_S{}.mat'.format(trial_number, str(speed).replace('.', ''))

    savemat(os.path.join(data_dir, filename), data_dict)


def zipdir(path, zip):
    for root, dirs, files in os.walk(path):
        for file in files:
            zip.write(os.path.join(root, file))

zipf = zipfile.ZipFile('nick.zip', 'w')
zipdir(data_dir, zipf)
zipf.close()
