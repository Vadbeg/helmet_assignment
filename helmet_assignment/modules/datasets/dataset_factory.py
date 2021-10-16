from __future__ import absolute_import, division, print_function

from helmet_assignment.modules.datasets.jde import JointDataset


def get_dataset(dataset, task):
    if task == 'mot':
        return JointDataset
    else:
        return None
