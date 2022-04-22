from utilities import *

DATA_PATH = "./Data_FD003/preprocessed data/"
attribute = ['Unit', 'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
             'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
df_train = pd.read_csv(DATA_PATH + 'TD_data.csv', names=attribute, header=None)
df_test = pd.read_csv(DATA_PATH + 'Test_data.csv', names=attribute, header=None)

df_train = Preprocessing.add_timeseries(df_train)
df_test = Preprocessing.add_timeseries(df_test)

train_label = pd.read_csv(DATA_PATH + "TD_mode.csv", header=None).values
test_label = pd.read_csv(DATA_PATH + "Test_mode.csv", header=None).values

train_dataset = AircraftDataset(df_train, train_label)  # 不插0计算创建dataset的子类
test_dataset = AircraftDataset(df_test, test_label)

train_dataset_mode_a = AircraftDataset_one_mode(df_train, train_label, -1)
train_dataset_mode_b = AircraftDataset_one_mode(df_train, train_label, 1)
test_dataset_mode_a = AircraftDataset_one_mode(df_test, test_label, -1)
test_dataset_mode_b = AircraftDataset_one_mode(df_test, test_label, 1)

path = DATA_PATH+'RUL.csv'
RUL_frame = pd.read_csv(path, header=None)
RUL = RUL_frame.values[:, 0]

index_a = np.where(test_label == -1)[0]
index_b = np.where(test_label == 1)[0]

def ln_get_hat_rho_lk_numerator_m(x_lm, Phi_l, *args):
    # [mu_mk,Sigma_mk,sigma_mk_2]
    mu_mk, Sigma_mk, sigma_mk_2 = args
    # return mu_mk,Sigma_mk,sigma_mk_2
    d = len(x_lm)  # dim
    mean = Phi_l @ mu_mk
    var = Phi_l @ Sigma_mk @ Phi_l.T + sigma_mk_2 * np.identity(d)
    ln_reg_part = (-1 / 2) * (d * np.log(2 * np.pi) + np.linalg.slogdet(var)[1])
    exp_part = (x_lm - mean).T @ var @ (x_lm - mean) / (-2)
    return (ln_reg_part + exp_part).item()


def ln_get_hat_rho_lk_numerator(instance_l, *args):
    # [pi,w,[mu,sigma],[mu,sigma]]
    dai_hao = ['T24', 'T30', 'T50', 'P30', 'Ps30', 'phi', 'W31', 'W32']
    # print(args[0][2])
    Phi_l = instance_l["Phi_l"]
    x_l = instance_l["input"]
    pi_k = args[0]
    w_k = args[1]
    sum_lk = 0
    # 单个sensor的对数和
    for idx, name in enumerate(dai_hao):
        args_m = args[idx + 2]
        # print(ln_get_hat_rho_lk_numerator_m(x_lm,Phi_l,*args_m))
        x_lm = instance_l[name]
        sum_lk += ln_get_hat_rho_lk_numerator_m(x_lm, Phi_l, *args_m)
    # 再加上HI的对数
    sum_lk += ln_get_hat_rho_lk_numerator_m(x_l @ w_k, Phi_l, *args[-1])
    return np.log(pi_k) + sum_lk  # pi_k乘上


def get_hat_rho_l(instance_l, *args
                  # [pi,w,[mu,sigma],[mu,sigma]],[pi,w,[mu,sigma],[mu,sigma]]
                  ):
    num_mode = len(args)
    ln_numerator_list = []
    for idx in range(num_mode):
        ln_numerator_list.append(ln_get_hat_rho_lk_numerator(instance_l, *args[idx]))
    sorted_list = np.sort(ln_numerator_list)[::-1]
    if np.abs(sorted_list[0] - sorted_list[-1]) > CFG.hat_rho_control:
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


def get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_lm, Phi_l, *args):
    # [mu_mk,Sigma_mk,sigma_mk_2]
    [mu_mk, Sigma_mk, sigma_mk_2] = args
    hat_Sigma_lm_k = np.linalg.inv(Phi_l.T @ Phi_l / sigma_mk_2 + np.linalg.inv(Sigma_mk))
    hat_Gamma_lm_k = hat_Sigma_lm_k @ (Phi_l.T @ x_lm / sigma_mk_2 + np.linalg.inv(Sigma_mk) @ mu_mk)
    return hat_Gamma_lm_k, hat_Sigma_lm_k


def get_new_para(instance, *args):
    num_mode = len(args)
    mu_list = []
    Sigma_list = []
    sigma_2_list = []
    dai_hao = ['T24', 'T30', 'T50', 'P30', 'Ps30', 'phi', 'W31', 'W32']
    pi_list = np.zeros((num_mode, 1))
    for idx in range(num_mode):
        sum_rho_lk = 0
        sum_rho_lk_nl = 0
        mu_k_list = np.zeros((len(dai_hao) + 1, 3, 1))  # xl1.xl2,xl3,yl
        # Sigma_k_list = np.zeros((len(dai_hao)+1,3,3))
        for l in tqdm(range(len(instance))):
            n_l = len(instance[l]["input"])
            hat_rho_lk = get_hat_rho_l(instance[l], *args)[idx]
            Phi_l = instance[l]["Phi_l"]
            x_l = instance[l]["input"]
            pi_k = args[idx][0]
            w_k = args[idx][1]
            # for x_lm

            for m, name in enumerate(dai_hao):
                args_m = args[idx][m + 2]
                mu_mk = args_m[idx][0]
                # print(ln_get_hat_rho_lk_numerator_m(x_lm,Phi_l,*args_m))
                x_lm = instance[l][name]
                hat_Gamma_lm_k, hat_Sigma_lm_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_lm, Phi_l,
                                                                                       *args_m)

                mu_k_list[m] += hat_rho_lk * hat_Gamma_lm_k
            #                 Sigma_k_list[m] += hat_rho_lk * (hat_Sigma_lm_k +
            #                                 (hat_Gamma_lm_k-mu_mk)@(hat_Gamma_lm_k-mu_mk).T)

            # for y
            mu_k = args[idx][-1][0]
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_l @ w_k, Phi_l,
                                                                                 *args[idx][-1])
            mu_k_list[-1] += hat_rho_lk * hat_Gamma_l_k
            #             Sigma_k_list[-1] += hat_rho_lk * (hat_Sigma_lm_k +
            #                                 (hat_Gamma_lm_k-mu_k)@(hat_Gamma_lm_k-mu_k).T)

            sum_rho_lk += hat_rho_lk
            sum_rho_lk_nl += hat_rho_lk * n_l

        for m, name in enumerate(dai_hao):
            mu_mk0 = CFG.mu_0_list[idx][m]
            mu_k_list[m] += CFG.beta * mu_mk0
        mu_k0 = CFG.mu_0_list[idx][-1]
        mu_k_list[-1] += CFG.beta * mu_k0
        mu_list.append(mu_k_list / (sum_rho_lk + CFG.beta))
        # Sigma_list.append(Sigma_k_list/sum_rho_lk)
    for idx in range(num_mode):
        sum_rho_lk = 0
        sum_rho_lk_nl = 0
        #         mu_k_list = np.zeros((len(dai_hao)+1,3,1)) # xl1.xl2,xl3,yl
        Sigma_k_list = np.zeros((len(dai_hao) + 1, 3, 3))
        sigma_2_k_list = np.zeros((len(dai_hao) + 1, 1, 1))
        for l in tqdm(range(len(instance))):
            n_l = len(instance[l]["input"])
            hat_rho_lk = get_hat_rho_l(instance[l], *args)[idx]
            Phi_l = instance[l]["Phi_l"]
            x_l = instance[l]["input"]
            pi_k = args[idx][0]
            w_k = args[idx][1]
            # for x_lm

            for m, name in enumerate(dai_hao):
                args_m = args[idx][m + 2]
                mu_mk = mu_list[idx][m]
                # print(ln_get_hat_rho_lk_numerator_m(x_lm,Phi_l,*args_m))
                x_lm = instance[l][name]
                hat_Gamma_lm_k, hat_Sigma_lm_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_lm, Phi_l,
                                                                                       *args_m)
                # mu_k_list[m] += hat_rho_lk * hat_Gamma_lm_k
                Sigma_k_list[m] += hat_rho_lk * (hat_Sigma_lm_k +
                                                 (hat_Gamma_lm_k - mu_mk) @ (hat_Gamma_lm_k - mu_mk).T)
                sigma_2_k_list[m] += hat_rho_lk * ((x_lm - Phi_l @ hat_Gamma_lm_k).T @
                                                   (x_lm - Phi_l @ hat_Gamma_lm_k) +
                                                   np.trace(Phi_l @ hat_Sigma_lm_k @ Phi_l.T)
                                                   )

            # for y
            mu_k = mu_list[idx][-1]
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_l @ w_k, Phi_l,
                                                                                 *args[idx][-1])

            # mu_k_list[-1] += hat_rho_lk * hat_Gamma_l_k
            Sigma_k_list[-1] += hat_rho_lk * (hat_Sigma_l_k +
                                              (hat_Gamma_l_k - mu_k) @ (hat_Gamma_l_k - mu_k).T)
            sigma_2_k_list[-1] += hat_rho_lk * ((x_l @ w_k - Phi_l @ hat_Gamma_l_k).T @
                                                (x_l @ w_k - Phi_l @ hat_Gamma_l_k) +
                                                np.trace(Phi_l @ hat_Sigma_l_k @ Phi_l.T)
                                                )

            sum_rho_lk += hat_rho_lk
            sum_rho_lk_nl += hat_rho_lk * n_l
        # mu_list.append(mu_k_list/sum_rho_lk)
        for m, name in enumerate(dai_hao):
            mu_mk0 = CFG.mu_0_list[idx][m]
            Sigma_mk0 = CFG.Sigma_0_list[idx][m]
            mu_mk = mu_list[idx][m]
            Sigma_k_list[m] += Sigma_mk0 + CFG.beta * (mu_mk-mu_mk0)@(mu_mk-mu_mk0).T
        mu_k0 = CFG.mu_0_list[idx][-1]
        Sigma_k0 = CFG.Sigma_0_list[idx][-1]
        mu_k = mu_list[idx][-1]
        Sigma_k_list[-1] += Sigma_k0 + CFG.beta * (mu_k-mu_k0)@(mu_k-mu_k0).T

        Sigma_list.append(Sigma_k_list / (sum_rho_lk + CFG.tau + CFG.dim + 2))
        sigma_2_list.append(sigma_2_k_list / sum_rho_lk_nl)
        pi_list[idx] = sum_rho_lk

    return pi_list / np.sum(pi_list), np.array(mu_list), np.array(Sigma_list), np.array(sigma_2_list)


# def get_hat_pi_k(instance, *args):
#     num_mode = len(args)
#     N_k_list = np.zeros((num_mode, 1))
#     for l in range(len(instance)):
#         N_k_list += get_hat_rho_l(instance[l], *args)
#     return N_k_list / np.sum(N_k_list)


def get_new_w(instance, Dk_list, *args):
    num_mode = len(args)
    w_list = []
    for idx in range(num_mode):
        sum_l_a2a2 = 0
        sum_l_a2b2 = 0
        pi_k = args[idx][0]
        w_k = args[idx][1]
        for l in range(len(instance)):
            [mu_k, Sigma_k, sigma_k_2] = args[idx][-1]
            x_l = instance[l]["input"]
            Phi_l = instance[l]["Phi_l"]
            Phi_tau_l = Phi_l[-1].reshape(1, 3)
            D_k = Dk_list[idx]
            hat_rho_lk = get_hat_rho_l(instance[l], *args)[idx]
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_l @ w_k, Phi_l,
                                                                                 *args[idx][-1])
            denominator = np.sqrt(Phi_tau_l @ Sigma_k @ Phi_l.T @ Phi_l @ hat_Sigma_l_k @ Phi_tau_l.T)
            # denominator = np.sqrt(Phi_tau_l @ hat_Sigma_l_k @ Phi_l.T @ Phi_l @ hat_Sigma_l_k @ Phi_tau_l.T)
            a2 = (Phi_tau_l @ hat_Sigma_l_k @ Phi_l.T @ x_l) / denominator  # 1*6
            b2 = sigma_k_2 * (D_k - Phi_tau_l @ hat_Sigma_l_k @ np.linalg.inv(Sigma_k) @ mu_k) / denominator  # 1*1

            sum_l_a2a2 += hat_rho_lk * a2.T @ a2  # 6*6
            sum_l_a2b2 += hat_rho_lk * a2.T * b2  # 6*1
        w_list.append(np.linalg.inv(sum_l_a2a2) @ sum_l_a2b2)
    return np.array(w_list)


def repack_para(pi_list, w_list, mu_list, Sigma_list, sigma_2_list):
    para = []
    for k in range(len(mu_list)):
        para_k = [pi_list[k].item(), w_list[k]]
        for m in range(len(mu_list[k])):
            para_k.append([mu_list[k][m], Sigma_list[k][m], sigma_2_list[k][m].item()])
        para.append(para_k)
    return para


def pack_para(*args):
    # [pik,wk,[],[]],[pik,wk,[],[]]
    pi_list = []
    w_list = []
    mu_list = []
    Sigma_list = []
    sigma_2_list = []
    num_mode = len(args)
    for idx in range(num_mode):
        pi_k = args[idx][0]
        w_k = args[idx][1]
        length = len(args[idx]) - 2
        mu_k_list = np.zeros((length, 3, 1))
        Sigma_k_list = np.zeros((length, 3, 3))
        sigma_2_k_list = np.zeros((length, 1, 1))
        for m in range(length):
            mu_k_list[m] = args[idx][m + 2][0]
            Sigma_k_list[m] = args[idx][m + 2][1]
            sigma_2_k_list[m] = args[idx][m + 2][2]
        pi_list.append(pi_k)
        w_list.append(w_k)
        mu_list.append(mu_k_list)
        Sigma_list.append(Sigma_k_list)
        sigma_2_list.append(sigma_2_k_list)
    return np.array(pi_list), np.array(w_list), np.array(mu_list), \
           np.array(Sigma_list), np.array(sigma_2_list)


def EM_iteration(instance, Dk_list, iteration, limitation, *args):
    epoch = 1
    loss_list = [1000, 100]
    while epoch < iteration:
        # get old para
        pi_list, w_list, mu_list, Sigma_list, sigma_2_list = pack_para(*args)
        # update w
        w_list_new = get_new_w(instance, Dk_list, *args)
        args_new_w = repack_para(pi_list, w_list_new, mu_list, Sigma_list, sigma_2_list)
        # update other
        pi_list_new, mu_list_new, Sigma_list_new, sigma_2_list_new = get_new_para(instance, *args_new_w)
        # pi_list_new = get_hat_pi_k(instance,*args_new_w)
        args_new_all = repack_para(pi_list_new, w_list_new, mu_list_new, Sigma_list_new, sigma_2_list_new)
        loss = rmse(mu_list, mu_list_new) + rmse(Sigma_list, Sigma_list_new) + rmse(sigma_2_list, sigma_2_list_new) + \
               rmse(pi_list, pi_list_new) + rmse(w_list, w_list_new)
        args = args_new_all
        loss_list.append(loss)
        if loss > limitation and (abs(loss_list[-1] - loss_list[-2]) > 0.005 or \
                                  abs(loss_list[-2] - loss_list[-3]) > 0.005):
            print(f"iteration:{epoch},loss:{loss}")
            epoch += 1
            continue
        else:
            print(f"Done! iteration:{epoch},loss:{loss}")
            return args
            break


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

    dk_list = [-10, 10]
    result_args_a = EM_iteration(train_dataset_mode_a, [-10], 100, 0.125, *args_a)
    result_args_b = EM_iteration(train_dataset_mode_b, [10], 100, 0.125, *args_b)
    args_all = result_args_a + result_args_b
    result_args = EM_iteration(train_dataset, dk_list, 100, 0.2, *args_all)
    np.save('./numpy/arg_all_result_pycharm.npy', np.array(result_args, dtype=object))
