import numpy as np

from utilities import *


def ln_get_hat_rho_k_numerator(x_l, Phi_l, pi_k, w_k, mu_k, Sigma_k, sigma_k_2):
    ln_hat_rho_k_numerator = 0
    for t in range(len(x_l)):
        post_mean = Phi_l[t] @ mu_k
        post_var = Phi_l[t] @ Sigma_k @ Phi_l[t].T + sigma_k_2
        exp_part = (x_l[t] @ w_k - post_mean) ** 2 / (-2 * post_var)
        reg_part = 2 * np.pi * post_var
        ln_PDF = np.log(1 / np.sqrt(reg_part)) + exp_part
        ln_hat_rho_k_numerator += ln_PDF
    return ln_hat_rho_k_numerator + np.log(pi_k)


# a=ln_get_hat_rho_k_numerator(pi_k, x_l, w_k, Phi_l, mu_k, Sigma_k, sigma_k_2)
# b=ln_get_hat_rho_k_numerator(pi_k, x_l, w_k, Phi_l, mu_k, Sigma_k, sigma_k_2)
# c=ln_get_hat_rho_k_numerator(pi_k, x_l, w_k, Phi_l, mu_k, Sigma_k, sigma_k_2)
#
# hat_rho_l_a=1/(np.exp(b-a)+np.exp(c-a)+1)

# def ln_get_hat_rho_k_numerator(x_l, Phi_l, pi_k, w_k, mu_k, Sigma_k, sigma_k_2):
#     k = len(x_l)  # dim
#     post_mean = Phi_l @ mu_k
#     post_var = Phi_l @ Sigma_k @ Phi_l.T + sigma_k_2 * np.identity(k)
#     ln_reg_part = (-1 / 2) * (k * np.log(2 * np.pi) + np.log(np.linalg.det(post_var)))
#     # 1 / np.sqrt(((2 * np.pi) ** k) * np.linalg.det(post_var))
#     exp_part = (x_l @ w_k - post_mean).T @ post_var @ (x_l @ w_k - post_mean) / (-2)
#     return np.log(pi_k) + ln_reg_part + exp_part


# def get_hat_rho_l_k_numerator(x_l, Phi_l, pi_k, w_k, mu_k, Sigma_k, sigma_k_2):
#     k = len(x_l)  # dim
#     post_mean = Phi_l @ mu_k
#     post_var = Phi_l @ Sigma_k @ Phi_l.T + sigma_k_2 * np.identity(k)
#     reg_part = ((2 * np.pi) ** k) * np.linalg.det(post_var)
#     exp_part = (x_l @ w_k - post_mean).T @ post_var @ (x_l @ w_k - post_mean) / (-2)
#     pdf = 1 / np.sqrt(reg_part) * np.exp(exp_part)
#     return pi_k * pdf


# def get_hat_rho_l(x_l, Phi_l, *arg
#                   #                   [pi_k, w_k,  mu_k, Sigma_k, sigma_k_2],
#                   #                      [pi_k, w_k, mu_k, Sigma_k, sigma_k_2]
#                   ):
#     num_mode = len(arg)
#     ln_numerator_list = []
#     for idx in range(num_mode):
#         [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
#         ln_numerator_list.append(ln_get_hat_rho_k_numerator(x_l, Phi_l, pi_k,
#                                                             w_k, mu_k,
#                                                             Sigma_k, sigma_k_2))
#     hat_rho_l_list = []
#     for idx in range(num_mode):
#         denominator = 0
#         for idx_ in range(num_mode):
#             denominator += np.exp(ln_numerator_list[idx_] - ln_numerator_list[idx])
#         hat_rho_l_list.append(1 / denominator.item())
#     return np.array(hat_rho_l_list).reshape(num_mode, 1)

def get_hat_rho_l(x_l, Phi_l, *arg
                  #                   [pi_k, w_k,  mu_k, Sigma_k, sigma_k_2],
                  #                      [pi_k, w_k, mu_k, Sigma_k, sigma_k_2]
                  ):
    num_mode = len(arg)
    ln_numerator_list = []
    for idx in range(num_mode):
        [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
        ln_numerator_list.append(ln_get_hat_rho_k_numerator(x_l, Phi_l, pi_k,
                                                            w_k, mu_k,
                                                            Sigma_k, sigma_k_2))
    sorted_list = np.sort(ln_numerator_list)[::-1]
    if np.abs(sorted_list[0] - sorted_list[-1]) > 10:
        max_idx = np.argmax(ln_numerator_list)
        hat_rho_l = np.zeros((num_mode, 1))
        hat_rho_l[max_idx] = 1
        return hat_rho_l
    else:
        hat_rho_l_list = []
        for idx in range(num_mode):
            denominator = 0
            for idx_ in range(num_mode):
                denominator += np.exp(ln_numerator_list[idx_] - ln_numerator_list[idx])
            hat_rho_l_list.append(1 / denominator.item())
        return np.array(hat_rho_l_list).reshape(num_mode, 1)


def get_hat_Gamma_l_k_and_hat_Sigma_l_k(x_l, Phi_l, w_k, mu_k, Sigma_k, sigma_k_2):
    hat_Sigma_l_k = np.linalg.inv(Phi_l.T @ Phi_l / sigma_k_2 + np.linalg.inv(Sigma_k))
    hat_Gamma_l_k = hat_Sigma_l_k @ (Phi_l.T @ x_l @ w_k / sigma_k_2 + np.linalg.inv(Sigma_k) @ mu_k)
    return hat_Gamma_l_k, hat_Sigma_l_k


def get_N_k_and_hat_pi_k(instance, *arg):
    num_mode = len(arg)
    N_k_list = np.zeros((num_mode, 1))
    for l in range(len(instance)):
        x_l = instance[l]["input"]
        Phi_l = instance[l]["Phi_l"]
        N_k_list += get_hat_rho_l(x_l, Phi_l, *arg)
    return N_k_list, N_k_list / np.sum(N_k_list)


def get_new_mu_list(instance, *arg):
    num_mode = len(arg)
    mu_list = []

    for idx in range(num_mode):
        N_k = 0
        sum_l_mu_k = 0

        for l in range(len(instance)):
            [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
            x_l = instance[l]["input"]
            Phi_l = instance[l]["Phi_l"]
            rho_l_k = get_hat_rho_l(x_l, Phi_l, *arg)[idx]
            N_k += rho_l_k
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_l_k_and_hat_Sigma_l_k(x_l, Phi_l, w_k, mu_k,
                                                                               Sigma_k, sigma_k_2)
            sum_l_mu_k += rho_l_k * hat_Gamma_l_k

        mu_list.append(sum_l_mu_k / N_k)
    return np.array(mu_list)


def get_new_Sigma_K_list(instance, *arg):
    """
    :param instance:
    :param arg:[pi_k, w_k, mu_k, Sigma_k, sigma_k_2]
    :return: sigma_k of k mode k=1,2,3,k
    """
    num_mode = len(arg)
    mu_list = get_new_mu_list(instance, *arg)
    Sigma_list = []
    for idx in range(num_mode):
        N_k = 0
        sum_l_Sigma_k = 0
        for l in range(len(instance)):
            [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
            x_l = instance[l]["input"]
            Phi_l = instance[l]["Phi_l"]
            rho_l_k = get_hat_rho_l(x_l, Phi_l, *arg)[idx]
            N_k += rho_l_k

            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_l_k_and_hat_Sigma_l_k(x_l, Phi_l, w_k, mu_k,
                                                                               Sigma_k, sigma_k_2)
            hat_mu_k = mu_list[idx]

            sum_l_Sigma_k += rho_l_k * (hat_Sigma_l_k +
                                        (hat_Gamma_l_k - hat_mu_k).T @ (hat_Gamma_l_k - hat_mu_k))
        Sigma_list.append(sum_l_Sigma_k / N_k)
    return np.array(Sigma_list)


def get_new_sigma_k_2_list(instance, *arg):
    num_mode = len(arg)
    sigma_k_2_list = []
    for idx in range(num_mode):
        N_k = 0
        sum_l_sigma_k_2 = 0
        for l in range(len(instance)):
            [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
            x_l = instance[l]["input"]
            Phi_l = instance[l]["Phi_l"]
            rho_l_k = get_hat_rho_l(x_l, Phi_l, *arg)[idx]
            N_k += rho_l_k

            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_l_k_and_hat_Sigma_l_k(x_l, Phi_l, w_k, mu_k,
                                                                               Sigma_k, sigma_k_2)

            sum_l_sigma_k_2 += rho_l_k * ((x_l @ w_k - Phi_l @ hat_Gamma_l_k).T @
                                          (x_l @ w_k - Phi_l @ hat_Gamma_l_k) +
                                          np.trace(Phi_l @ hat_Sigma_l_k @ Phi_l.T))
        sigma_k_2_list.append(sum_l_sigma_k_2 / N_k)
    return np.array(sigma_k_2_list).reshape(num_mode, 1)


def get_new_w_k_list(instance, *arg):
    num_mode = len(arg)
    w_k_list = []
    for idx in range(num_mode):
        sum_l_a2a2 = 0
        sum_l_a2b2 = 0
        for l in range(len(instance)):
            [pi_k, w_k, mu_k, Sigma_k, sigma_k_2] = arg[idx]
            x_l = instance[l]["input"]
            Phi_l = instance[l]["Phi_l"]
            Phi_tau_l = Phi_l[-1].reshape(1, 3)
            D_k = 1
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_l_k_and_hat_Sigma_l_k(x_l, Phi_l, w_k, mu_k,
                                                                               Sigma_k, sigma_k_2)
            denominator = np.sqrt(Phi_tau_l @ Sigma_k @ Phi_l.T @ Phi_l @ hat_Sigma_l_k @ Phi_tau_l.T)
            a2 = (Phi_tau_l @ hat_Sigma_l_k @ Phi_l.T @ x_l) / denominator  # 1*6
            b2 = sigma_k_2 * (D_k - Phi_tau_l @ hat_Sigma_l_k @ np.linalg.inv(Sigma_k) @ mu_k) / denominator  # 1*1

            sum_l_a2a2 += a2.T @ a2  # 6*6
            sum_l_a2b2 += a2.T * b2  # 6*1
        w_k_list.append(np.linalg.inv(sum_l_a2a2) @ sum_l_a2b2)
    return np.array(w_k_list)


def EM_iteration(instance, iteration=100, limitation=0.01, *arg):
    num_mode = len(arg)
    itr = 0

    while itr < iteration:
        old_w = []
        old_mu = []
        old_Sigma = []
        old_sigma_2 = []
        for idx in range(num_mode):
            old_w.append(arg[idx][1])
            old_mu.append(arg[idx][2])
            old_Sigma.append(arg[idx][3])
            old_sigma_2.append(arg[idx][4])
        old_w = np.array(old_w)
        old_mu = np.array(old_mu)
        old_Sigma = np.array(old_Sigma)
        old_sigma_2 = np.array(old_sigma_2)

        updated_w = get_new_w_k_list(instance, *arg)
        for idx in range(num_mode):
            arg[idx][1] = updated_w[idx]

        updated_pi_k = get_N_k_and_hat_pi_k(instance, *arg)[1]
        updated_mu = get_new_mu_list(instance, *arg)  # num_mode*3*1
        updated_Sigma = get_new_Sigma_K_list(instance, *arg)
        updated_sigma_2 = get_new_sigma_k_2_list(instance, *arg)

        loss_w = (np.square(old_w - updated_w)).mean().item()
        loss_mu = (np.square(old_mu - updated_mu)).mean().item()
        loss_Sigma = (np.square(old_Sigma - updated_Sigma)).mean().item()
        loss_sigma_2 = (np.square(old_sigma_2 - updated_sigma_2)).mean().item()

        for idx in range(num_mode):
            arg[idx][0] = updated_pi_k[idx]
            arg[idx][2] = updated_mu[idx]
            arg[idx][3] = updated_Sigma[idx]
            arg[idx][4] = updated_sigma_2[idx]

        if loss_w < limitation and loss_w < limitation and loss_Sigma < limitation and loss_sigma_2 < limitation:
            print(f"converge!, itr={itr}")
            break
        else:
            itr += 1
            if itr % 10 == 0:
                loss = loss_w + loss_mu + loss_Sigma + loss_sigma_2
                print(f"itr={itr},loss={loss}")
        if itr == iteration:
            print(f"g!, itr={itr}")
            break
    return arg
