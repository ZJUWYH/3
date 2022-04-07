import numpy as np

from EM_part import *

# args_result = np.load('./numpy/arg_all_result0407.npy',allow_pickle=True).tolist()
# failure mode
path = DATA_PATH + 'RUL.csv'
RUL_frame = pd.read_csv(path, header=None)
RUL = RUL_frame.values[:, 0]


def mode_para_selection(instance_q, dk_list, *args):
    """
    *args [pi,w,[mu,sigma],[mu,sigma]],[pi,w,[mu,sigma],[mu,sigma]]
    return [pi,w,mu,...]
    """
    mode_q = np.argmax(get_hat_rho_l(instance_q, *args))
    selected_args = []
    args_q = args[mode_q]
    selected_args.append(dk_list[mode_q])  # dk
    selected_args.append(args_q[1])  # wk
    # selected_args.appned(args_q[1])
    for idx in range(3):
        selected_args.append(args_q[-1][idx])
    return selected_args


def get_post_Sigma_q_and_mu_q(instance_q, *args):
    """
    args:select_args,[pi,w,mu,...]
    """
    [hat_D, w, mu_Gamma, C_Gamma, sigma_2] = args
    x_q = instance_q["input"]
    Phi_q = instance_q["Phi_l"]
    Sigma_q = np.linalg.inv((Phi_q.T @ Phi_q) / sigma_2 + np.linalg.inv(C_Gamma))
    mu_q = Sigma_q @ ((Phi_q.T @ x_q @ w) / sigma_2 + np.linalg.inv(C_Gamma) @ mu_Gamma)
    return mu_q, Sigma_q


def get_CDF_t(instance_q, t, *args):
    n_q = len(instance_q["input"])
    Phi_q = instance_q["Phi_l"]
    [hat_D, w, mu_Gamma, C_Gamma, sigma_2] = args
    t_nl_t = n_q + t
    Phi_t_nl_t = np.array([1, t_nl_t / 500, (t_nl_t / 500) * (t_nl_t / 500)],
                          dtype=np.float64).reshape(1, 3)
    mu_q, Sigma_q = get_post_Sigma_q_and_mu_q(instance_q, *args)
    mu_t_nl_t_0 = Phi_q[-1] @ mu_q
    mu_t_nl_t = Phi_t_nl_t @ mu_q
    sigma_t_nl_t = Phi_t_nl_t @ Sigma_q @ Phi_t_nl_t.T
    if mu_t_nl_t_0 < hat_D:
        A = ((hat_D - mu_t_nl_t) / (np.sqrt(sigma_t_nl_t))).item()
    else:
        A = ((-hat_D + mu_t_nl_t) / (np.sqrt(sigma_t_nl_t))).item()
    Phi_t = 1 - scipy.stats.norm.cdf(A)
    return Phi_t


def get_single_RUL(instance_q, dk_list, *args):
    """
    *args:[pi,w,[mu,sigma],[mu,sigma]],[pi,w,[mu,sigma],[mu,sigma]] previous results
    """
    select_args = mode_para_selection(instance_q, dk_list, *args)
    box = []
    for t in range(500):
        A = get_CDF_t(instance_q, t, *select_args)
        B = get_CDF_t(instance_q, 0, *select_args)
        C = (A - B) / (1 - B)
        D = abs(C - 0.5)
        box.append(D)
    RUL_q = box.index(min(box)) + 1
    return RUL_q


def get_RUL(instance, dk_list, *args):
    length = len(instance)
    RUL = []
    for idx in tqdm(range(length)):
        RUL_q = get_single_RUL(instance[idx], dk_list, *args)
        RUL.append(RUL_q)
    return np.array(RUL)


if __name__ == "__main__":
    # initial
    set_seed(612)
    w_k = np.random.rand(6, 1)
    pi_k = np.random.random(size=None)
    mu_mk = np.random.rand(3, 1)
    Sigma_mk = generate_rand_psd(3)
    sigma_mk_2 = np.random.random(size=None)

    w_j = np.random.rand(6, 1)
    pi_j = np.random.random(size=None)
    mu_mj = np.random.rand(3, 1)
    Sigma_mj = generate_rand_psd(3)
    sigma_mj_2 = np.random.random(size=None)

    args_a = [[pi_k, w_k, [mu_mk, Sigma_mk, sigma_mk_2],
               [mu_mk, Sigma_mk, sigma_mk_2], [mu_mk, Sigma_mk, sigma_mk_2], [mu_mk, Sigma_mk, sigma_mk_2],
               [mu_mk, Sigma_mk, sigma_mk_2], [mu_mk, Sigma_mk, sigma_mk_2], [mu_mk, Sigma_mk, sigma_mk_2]]]
    args_b = [[pi_k, w_k, [mu_mj, Sigma_mj, sigma_mj_2],
               [mu_mj, Sigma_mj, sigma_mj_2], [mu_mj, Sigma_mj, sigma_mj_2], [mu_mj, Sigma_mj, sigma_mj_2],
               [mu_mj, Sigma_mj, sigma_mj_2], [mu_mj, Sigma_mj, sigma_mj_2], [mu_mj, Sigma_mj, sigma_mj_2]]]

    dk_list = [10,20]

    #EM
    result_args_a = EM_iteration(train_dataset_mode_a, [10], 100, 0.125, *args_a)
    result_args_b = EM_iteration(train_dataset_mode_b, [20], 100, 0.125, *args_b)
    args_all = result_args_a + result_args_b
    result_args = EM_iteration(train_dataset, dk_list, 100, 0.2, *args_all)
    np.save('./numpy/arg_all_result_pycharm.npy', np.array(result_args, dtype=object))

    #RUL
    args_result = np.load('./numpy/arg_all_result_pycharm.npy', allow_pickle=True).tolist()
    RUL_estimate = get_RUL(test_dataset, dk_list, *args_result)
    print(np.sum(np.abs(RUL-RUL_estimate)))
