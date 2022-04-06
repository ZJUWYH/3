import numpy as np

from utilities import *
from EM_part_discard import *


# failure mode

def mode_recognition_for_in_unit_q(instance, idx, *new_arg):
    """
    :param instance: test_dataset
    :param idx: idx
    :param new_arg: the updated arg
    :return: the [pi_k, w_k,  mu_k, Sigma_k, sigma_k_2] of the recognized mode
    """
    x_q = instance[idx]["input"]
    Phi_q = instance[idx]["Phi_l"]
    mode = np.argmax(get_hat_rho_l(x_q, Phi_q, *new_arg))

