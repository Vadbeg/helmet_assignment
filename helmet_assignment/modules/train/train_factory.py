from __future__ import absolute_import, division, print_function

from .mot import MotTrainer

train_factory = {
    'mot': MotTrainer,
}
