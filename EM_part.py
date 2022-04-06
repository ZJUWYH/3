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


def ln_get_hat_rho_lk_numerator_m(x_lm, Phi_l, *args):
    # [mu_mk,Sigma_mk,sigma_mk_2]
    mu_mk, Sigma_mk, sigma_mk_2 = args
    # return mu_mk,Sigma_mk,sigma_mk_2
    d = len(x_lm)  # dim
    mean = Phi_l @ mu_mk
    var = Phi_l @ Sigma_mk @ Phi_l.T + sigma_mk_2 * np.identity(d)
    ln_reg_part = (-1 / 2) * (d * np.log(2 * np.pi) + np.log(np.linalg.det(var)))
    exp_part = (x_lm - mean).T @ var @ (x_lm - mean) / (-2)
    return (ln_reg_part + exp_part).item()


def ln_get_hat_rho_lk_numerator(instance_l, *args):
    # [pi,w,[mu,sigma],[mu,sigma]]
    dai_hao = ["s1", "s2", "s3", "s4", "s5", "s6"]
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


def get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_lm, Phi_l, *args):
    # [mu_mk,Sigma_mk,sigma_mk_2]
    mu_mk, Sigma_mk, sigma_mk_2 = args
    hat_Sigma_lm_k = np.linalg.inv(Phi_l.T @ Phi_l / sigma_mk_2 + np.linalg.inv(Sigma_mk))
    hat_Gamma_lm_k = hat_Sigma_lm_k @ (Phi_l.T @ x_lm / sigma_mk_2 + np.linalg.inv(Sigma_mk) @ mu_mk)
    return hat_Gamma_lm_k, hat_Sigma_lm_k


def get_new_para(instance, *args):
    num_mode = len(args)
    mu_list = []
    Sigma_list = []
    sigma_2_list = []
    dai_hao = ["s1", "s2", "s3", "s4", "s5", "s6"]
    for idx in range(num_mode):
        sum_rho_lk = 0
        sum_rho_lk_nl = 0
        mu_k_list = np.zeros((len(dai_hao) + 1, 3, 1))  # xl1.xl2,xl3,yl
        # Sigma_k_list = np.zeros((len(dai_hao)+1,3,3))
        for l in range(len(instance)):
            n_l = len(instance[l]["input"])
            hat_rho_lk = get_hat_rho_l(instance[l], *args)[idx]
            Phi_l = instance[l]["Phi_l"]
            x_l = instance[l]["input"]
            pi_k = args[idx][0]
            w_k = args[idx][1]
            # for x_lm

            for m, name in enumerate(dai_hao):
                args_m = args[idx][m + 2]
                # mu_mk = args_m[idx][0]
                # print(ln_get_hat_rho_lk_numerator_m(x_lm,Phi_l,*args_m))
                x_lm = instance[l][name]
                hat_Gamma_lm_k, hat_Sigma_lm_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_lm, Phi_l,
                                                                                       *args_m)
                mu_k_list[m] += hat_rho_lk * hat_Gamma_lm_k
            #                 Sigma_k_list[m] += hat_rho_lk * (hat_Sigma_lm_k +
            #                                 (hat_Gamma_lm_k-mu_mk)@(hat_Gamma_lm_k-mu_mk).T)

            # for y
            # mu_k = args[idx][-1][0]
            hat_Gamma_l_k, hat_Sigma_l_k = get_hat_Gamma_lm_k_and_hat_Sigma_lm_k(x_l @ w_k, Phi_l,
                                                                                 *args[idx][-1])

            mu_k_list[-1] += hat_rho_lk * hat_Gamma_l_k
            #             Sigma_k_list[-1] += hat_rho_lk * (hat_Sigma_lm_k +
            #                                 (hat_Gamma_lm_k-mu_k)@(hat_Gamma_lm_k-mu_k).T)

            sum_rho_lk += hat_rho_lk
            sum_rho_lk_nl += hat_rho_lk * n_l
        mu_list.append(mu_k_list / sum_rho_lk)
        # Sigma_list.append(Sigma_k_list/sum_rho_lk)
    for idx in range(num_mode):
        #         sum_rho_lk = 0
        #         sum_rho_lk_nl = 0
        #         mu_k_list = np.zeros((len(dai_hao)+1,3,1)) # xl1.xl2,xl3,yl
        Sigma_k_list = np.zeros((len(dai_hao) + 1, 3, 3))
        sigma_2_k_list = np.zeros((len(dai_hao) + 1, 1, 1))
        for l in range(len(instance)):
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

        #             sum_rho_lk += hat_rho_lk
        #             sum_rho_lk_nl += hat_rho_lk * n_l
        # mu_list.append(mu_k_list/sum_rho_lk)
        Sigma_list.append(Sigma_k_list / sum_rho_lk)
        sigma_2_list.append(sigma_2_k_list / sum_rho_lk_nl)

    return mu_list, Sigma_list, sigma_2_list


def get_hat_pi_k(instance, *args):
    num_mode = len(args)
    N_k_list = np.zeros((num_mode, 1))
    for l in range(len(instance)):
        N_k_list += get_hat_rho_l(instance[l], *args)
    return N_k_list / np.sum(N_k_list)


def repack_para(pi_list, w_list, mu_list, Sigma_list, sigma_2_list):
    para = []
    for k in range(len(mu_list)):
        para_k = [pi_list[k].item(), w_list[k]]
        for m in range(len(mu_list[k])):
            para_k.append([mu_list[k][m], Sigma_list[k][m], sigma_2_list[k][m].item()])
        para.append(para_k)
    return para
