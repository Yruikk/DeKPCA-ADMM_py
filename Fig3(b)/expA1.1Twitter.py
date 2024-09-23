import mpi4py.MPI as MPI
import math
import numpy as np
from numpy.linalg import multi_dot, eig
from func import gen_adjmat, centralize_kernel, solve_global_svd, my_inv, subspace
from scipy.linalg import orth, norm
from scipy.io import loadmat
import csv
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
import PGD


class MyPms:
    def __init__(self):
        self.worker_num = 10  # J in paper
        self.k = 50
        self.target_k = 1
        self.train_num = 200
        self.test_num = 0
        self.ill_thres = 0.01
        self.max_repeat = 1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pms = MyPms()
nei_num = 2
file_path = 'Twitter.mat'

pms.local_n = pms.train_num + pms.test_num

# 1.1 Calculate adjacent matrix.
if rank == 0:
    adj_mat = gen_adjmat(pms.worker_num, nei_num)
else:
    adj_mat = None

adj_mat = comm.bcast(adj_mat, root=0)

# 1.2 Prepare and broadcast N_j.
nei = np.where(adj_mat[:, rank] == 1)
tmp = nei[0].reshape(-1, 1)
N_j = np.row_stack(([rank], tmp))  # Put j to the first.
norm_N_j = np.size(N_j)

# 1.3 Create new communicators based on N_j.
tmp = N_j.flatten()
a = tmp.tolist()
j_list = comm.allgather(a)
comm_j = [[] for i in range(pms.worker_num)]
for i in range(pms.worker_num):
    comm_j[i] = comm.Create(comm.group.Incl(j_list[i]))

N_j_nei = [[] for i in range(pms.worker_num)]
for i in range(pms.worker_num):
    if rank in j_list[i]:
        N_j_nei[i] = comm_j[i].gather(N_j, root=0)

# 1.4 Start repeating the experiment.
lam_ini = 0
lam_DeKPCA = 0
lam_JMLR = 0

acc_ini = np.zeros((1, pms.max_repeat))
acc_DeKPCA = np.zeros((1, pms.max_repeat))
acc_JMLR = np.zeros((1, pms.max_repeat))
if rank == 0:
    time_ini = np.zeros((1, pms.max_repeat))
    time_DeKPCA = np.zeros((1, pms.max_repeat))
    time_JMLR = np.zeros((1, pms.max_repeat))
else:
    time_ini = None
    time_DeKPCA = None
    time_JMLR = None

for rep_iter in range(pms.max_repeat):
    # -----(Start) Data Generation-----
    if rank == 0:
        data_all = loadmat(file_path, mat_dtype=True)
        XX = data_all['X']
        YY = data_all['Y']
        stack = np.column_stack((XX, YY))
        num_train = pms.worker_num * pms.train_num
        num_test = pms.test_num
        np.random.shuffle(stack)

        data_X = stack[:, 0:-1]
        data_Y = stack[:, -1]
        data_Y = data_Y.reshape(-1, 1)

        X_tr = data_X[0:num_train, :]
        Y_tr = data_Y[0:num_train, :]
        # X_te = data_X[num_train:num_train + num_test, :]
        # Y_te = data_Y[num_train:num_train + num_test, :]

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        X_train = min_max_scaler.fit_transform(X_tr)
        Y_train = preprocessing.scale(Y_tr)
        # X_test = min_max_scaler.fit_transform(X_te)
        # Y_test = preprocessing.scale(Y_te)

        split_X_tr = np.array_split(X_train, pms.worker_num, axis=0)
        split_Y_tr = np.array_split(Y_train, pms.worker_num, axis=0)
    else:
        split_X_tr = None
        split_Y_tr = None
        # X_test = None
        # Y_test = None
    X_tr_j = comm.scatter(split_X_tr, root=0)  # data: N x d
    X_tr_j = X_tr_j.T  # data: d x N
    X_tr_j = X_tr_j.astype(np.float64)
    Y_tr_j = comm.scatter(split_Y_tr, root=0)  # data: N x d
    Y_tr_j = Y_tr_j.T  # data: d x N
    Y_tr_j = Y_tr_j.astype(np.float64)
    # X_te_j = comm.bcast(X_test, root=0)  # data: N x d
    # X_te_j = X_te_j.T  # data: d x N
    # X_te_j = X_te_j.astype(np.float64)
    # Y_te_j = comm.bcast(Y_test, root=0)  # data: N x d
    # Y_te_j = Y_te_j.T  # data: d x N
    # Y_te_j = Y_te_j.astype(np.float64)

    data_j = np.array(X_tr_j)
    # data_j = np.column_stack((X_tr_j, X_te_j))

    local_n_nei = [[] for i in range(pms.worker_num)]
    for i in range(pms.worker_num):
        if rank in j_list[i]:
            local_n_nei[i] = comm_j[i].gather(pms.local_n, root=0)
    data_nei = [[] for i in range(pms.worker_num)]
    for i in range(pms.worker_num):
        if rank in j_list[i]:
            data_nei[i] = comm_j[i].gather(data_j, root=0)
    # -----(End) Data Generation-----
    pms.n = comm.allreduce(pms.local_n, op=MPI.SUM)
    pms.m = np.size(data_j, 0)
    pms.sigma = np.sqrt(pms.m)
    # -----(Start) Ground-turth-----
    data = comm.gather(data_j, root=0)  # 数据放在了一起
    if rank == 0:
        data_total = data[0]
        for i in range(pms.worker_num - 1):
            data_total = np.column_stack((data_total, data[i + 1]))  # data_total: d x N
        tic = MPI.Wtime()
        kernel_tt = rbf_kernel(data_total.T, data_total.T, 1 / pow(pms.sigma, 2))
        kernel_tt = centralize_kernel(kernel_tt)
        alpha_ggt, _ = solve_global_svd(kernel_tt, pms.target_k)
        toc = MPI.Wtime()
        time_ini[0, rep_iter] = toc - tic
        print('Single machine running time: ', toc - tic)
    else:
        data_total = None
        alpha_ggt = None
        kernel_tt = None
    data_total = comm.bcast(data_total, root=0)
    alpha_ggt = comm.bcast(alpha_ggt, root=0)
    kernel_tt = comm.bcast(kernel_tt, root=0)

    kernel_total_j = rbf_kernel(data_j.T, data_total.T, 1 / pow(pms.sigma, 2))
    kernel_total_j = centralize_kernel(kernel_total_j)
    # -----(End) Ground-turth-----

    # -----(Start) Kernel matrix preapre-----
    K_mat_j = [[[] for i in range(pms.worker_num)] for j in range(pms.worker_num)]
    for nei_iter_1 in range(norm_N_j):
        index_1 = N_j[nei_iter_1, 0]
        for nei_iter_2 in range(norm_N_j):
            index_2 = N_j[nei_iter_2, 0]

            data_1 = data_nei[rank][nei_iter_1]
            data_2 = data_nei[rank][nei_iter_2]
            tmp_kernel = rbf_kernel(data_1.T, data_2.T, 1 / pow(pms.sigma, 2))
            K_mat_j[index_1][index_2] = centralize_kernel(tmp_kernel)
    alpha_ini, lam_ini = solve_global_svd(K_mat_j[rank][rank], pms.target_k)

    K_mat_j[rank][rank] = K_mat_j[rank][rank] + pms.ill_thres * np.min(lam_ini) / pms.local_n * np.ones(
        (np.size(K_mat_j[rank][rank], 0), np.size(K_mat_j[rank][rank], 1)))
    inv_K_mat_j = [[] for i in range(pms.worker_num)]
    for nei_iter in range(norm_N_j):
        nei_tmp = N_j[nei_iter, 0]
        inv_K_mat_j[nei_tmp] = my_inv(K_mat_j[nei_tmp][nei_tmp])
    kernel_inv = my_inv(K_mat_j[rank][rank])
    alphg_gt = multi_dot([kernel_inv, kernel_total_j, alpha_ggt])

    acc_ini[0, rep_iter] = 0
    for i in range(pms.target_k):
        v1 = alpha_ini[:, i].reshape(-1, 1)
        v2 = alpha_ggt[:, i].reshape(-1, 1)
        acc_ini[0, rep_iter] += np.abs(
            multi_dot([v1.T, kernel_total_j, v2]) / np.sqrt(multi_dot([v1.T, K_mat_j[rank][rank], v1])) / np.sqrt(
                multi_dot([v2.T, kernel_tt, v2])))
    # -----(End) Kernel matrix preapre-----

    # -----(Start) DeKPCA-----
    # Input: K_mat_j + kernel_inv + N_j + pms
    # DeKPCA.1 Prepare auxiliary matrices(e_j, E_j) and \xi_j based on N_j.
    ee_j = np.zeros((pms.worker_num, 1))
    ee_j[rank] = 1
    ee = comm.allgather(ee_j)
    E_j = np.ones((1, norm_N_j))
    E = comm.allgather(E_j)

    xi_j = np.zeros((pms.worker_num, norm_N_j))
    for i in range(norm_N_j):
        xi_j[:, i] = ee[N_j[i, 0]].flatten()
    xi = comm.allgather(xi_j)

    adjust = np.zeros((pms.local_n, pms.local_n))
    alpha_total = np.zeros((pms.local_n, pms.target_k))

    tic = MPI.Wtime()
    phi_z = [[] for i in range(pms.worker_num)]
    phi_z_old = [[] for i in range(pms.worker_num)]
    for k_iter in range(pms.target_k):
        # DeKPCA.2 Initialize \eta_j = 0. Initialize \alpha_j with local data.
        eta_j = np.zeros((pms.local_n, norm_N_j))
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            phi_z_old[nei_tmp] = np.zeros((pms.local_n, 1))
        tmp = (K_mat_j[rank][rank] - adjust) @ (K_mat_j[rank][rank] - adjust)
        alpha_j, first_eig = solve_global_svd(tmp, 1)

        # DeKPCA.3 ADMM START
        stage = 1
        update_flag = 1
        update_thres = 1e-3
        stop_flag = 1e-3
        alpha_flag = 1
        z_flag = 1

        for ADMM_iter in range(50):
            # Increase rho with iterations.
            rho_j = 1000
            if stage == 1:
                rho_l = rho_j / 100
            elif stage == 2:
                rho_l = rho_j / 20
            else:
                rho_l = rho_j
            rho = np.zeros((1, norm_N_j))
            for nei_iter in range(norm_N_j):
                nei_tmp = N_j[nei_iter, 0]
                if nei_tmp == rank:
                    rho[0][nei_iter] = rho_j * first_eig
                else:
                    rho[0][nei_iter] = rho_l * first_eig
            H_hat = 1 / np.sum(rho)
            rho_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    rho_nei[i] = comm_j[i].gather(rho, root=0)

            # Broadcast \alpha and \eta.
            alpha_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    alpha_nei[i] = comm_j[i].gather(alpha_j, root=0)
            eta_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    eta_nei[i] = comm_j[i].gather(eta_j, root=0)

            # DeKPCA.3.1.1 Calculate Z.
            # (*)define phi_z[l] = \phi(X_l)^T * Z * e_j (\subset R^{local_l x 1}).
            if update_flag > update_thres:
                for nei_iter_1 in range(norm_N_j):
                    nei_tmp_1 = N_j[nei_iter_1, 0]
                    phi_z[nei_tmp_1] = np.zeros((local_n_nei[rank][nei_iter_1], 1))
                    for nei_iter_2 in range(norm_N_j):
                        nei_tmp_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[rank][nei_iter_2] == rank)
                        # (*)define tmp_z_l = K_l^{-1} * eta_l + alpha_l * E * rho (\subset R^{local_l x |N_l|}).
                        tmp_z_l = H_hat * multi_dot([K_mat_j[nei_tmp_1][nei_tmp_2], inv_K_mat_j[nei_tmp_2],
                                                     eta_nei[rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1)]) \
                                  + H_hat * rho_nei[rank][nei_iter_2][0][idx_z[0][0]] * np.dot(
                            K_mat_j[nei_tmp_1][nei_tmp_2],
                            alpha_nei[rank][nei_iter_2])
                        phi_z[nei_tmp_1] = phi_z[nei_tmp_1] + tmp_z_l

            # DeKPCA.3.1.2 Calculate ||Ze_j|| and project z_j.
            if update_flag > update_thres:
                z_norm = 1
                for nei_iter_1 in range(norm_N_j):
                    index_1 = N_j[nei_iter_1, 0]
                    idx_z = np.where(N_j_nei[rank][nei_iter_1] == rank)
                    inf_nei1 = np.dot(inv_K_mat_j[index_1], eta_nei[rank][nei_iter_1][:, idx_z[0][0]].reshape(-1, 1)) \
                               + rho_nei[rank][nei_iter_1][0][idx_z[0][0]] * alpha_nei[rank][nei_iter_1]
                    inf_nei1 = H_hat * inf_nei1
                    for nei_iter_2 in range(norm_N_j):
                        index_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[rank][nei_iter_2] == rank)
                        inf_nei2 = np.dot(inv_K_mat_j[index_2],
                                          eta_nei[rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1)) + \
                                   rho_nei[rank][nei_iter_2][0][idx_z[0][0]] * alpha_nei[rank][nei_iter_2]
                        inf_nei2 = H_hat * inf_nei2
                        z_norm += multi_dot([inf_nei1.T, K_mat_j[index_1][index_2], inf_nei2])[0][0]
                if z_norm > 1:
                    for nei_iter in range(norm_N_j):
                        nei_tmp = N_j[nei_iter, 0]
                        phi_z[nei_tmp] = phi_z[nei_tmp] / np.sqrt(z_norm)

            # DeKPCA.3.1.3 Build \phi(X_j) * Z * \xi_j
            if update_flag > update_thres:
                xx = [[] for i in range(pms.worker_num)]
                for i in range(pms.worker_num):
                    tmp_data = phi_z[i]
                    if rank in j_list[i]:
                        xx[i] = comm_j[i].gather(tmp_data, root=0)
                for i in range(norm_N_j):
                    if i == 0:
                        phi_z_xi = xx[rank][i]
                    else:
                        phi_z_xi = np.column_stack((phi_z_xi, xx[rank][i]))

            # DeKPCA.3.1.4 Align phi_z.
            if update_flag > update_thres:
                for nei_iter in range(norm_N_j):
                    if nei_iter != 0:
                        direction = np.dot(phi_z_xi[:, nei_iter].T, phi_z_xi[:, 0])
                        if direction < 0:
                            phi_z_xi[:, nei_iter] = -phi_z_xi[:, nei_iter]

            # DeKPCA.3.1.5 Compute the convergence of z.
            delta_z = 0
            for nei_iter in range(norm_N_j):
                nei_tmp = N_j[nei_iter, 0]
                delta_z += norm(phi_z[nei_tmp] - phi_z_old[nei_tmp])
            sum_delta_z = comm.allreduce(delta_z, op=MPI.SUM)
            if sum_delta_z < pms.worker_num * update_thres:
                z_flag = 1
            phi_z_old = np.array(phi_z, dtype=object)

            # Compute the update flag of z.
            A = np.dot(alpha_j, E_j)
            B = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten())])
            update_flag = np.sin(subspace(A, B))

            # DeKPCA.3.2 Update \alpha_j.
            # (*)define \alpha_j = H_alpha^-1 * f_alpha
            delta_alpha = 0
            alpha_old = np.array(alpha_j)
            H_alpha = -2 * np.dot(K_mat_j[rank][rank] - adjust, K_mat_j[rank][rank] - adjust) \
                      + multi_dot([E_j, np.diag(rho.flatten()), E_j.T])[0][0] * np.eye(pms.local_n)

            f_alpha = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten()), E_j.T]) - np.dot(eta_j, E_j.T)

            tmp_a = np.array(alpha_j)
            for ii in range(10):
                grad = H_alpha @ tmp_a - f_alpha  # calculate gradient
                lr = (grad.T @ grad) / (grad.T @ H_alpha @ grad)
                tmp_a_new = tmp_a - lr * grad
                check1 = norm(lr * grad)
                check2 = 0.5 * tmp_a.T @ H_alpha @ tmp_a - f_alpha.T @ tmp_a
                check3 = 0.5 * tmp_a_new.T @ H_alpha @ tmp_a_new - f_alpha.T @ tmp_a_new
                if check1 < 1e-3 or check2 < check3:
                    break
                tmp_a = np.array(tmp_a_new)
            alpha_j = np.array(tmp_a)
            delta_alpha += norm(alpha_j - alpha_old)
            sum_delta_alpha = comm.allreduce(delta_alpha, op=MPI.SUM)
            if sum_delta_alpha < pms.worker_num * update_thres:  # if alpha converges, then update eta.
                alpha_flag = 1

            # DeKPCA.3.3 Update and boardcast \eta_j.
            if z_flag == 1 and alpha_flag == 1:
                eta_old = np.array(eta_j)
                eta_j = eta_j + multi_dot([alpha_j, E_j, np.diag(rho.flatten())]) \
                        - multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten())])

                A = np.dot(alpha_j, E_j)
                B = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten())])
                update_flag = np.sin(subspace(A, B))
                sum_update_flag = comm.allreduce(update_flag, op=MPI.SUM)
                if sum_update_flag < pms.worker_num * stop_flag:
                    stage += 1
                    if stage == 4:
                        if rank == 0:
                            print("In iter", ADMM_iter + 1, "stage reaches 4.")
                        break
                alpha_flag = 0
                z_flag = 0
        alpha_j = alpha_j / np.sqrt(alpha_j.T @ K_mat_j[rank][rank] @ alpha_j)
        alpha_total[:, k_iter] = alpha_j.flatten()
        adjust = K_mat_j[rank][rank] @ alpha_total @ alpha_total.T @ K_mat_j[rank][rank].T

    toc = MPI.Wtime()
    time_each = toc - tic
    time_all = comm.allreduce(time_each, op=MPI.MAX)
    if rank == 0:
        time_DeKPCA[0, rep_iter] = time_all
        print('DeKPCA running time: ', time_all, '\n')
    alpha_DeKPCA_j = np.real(alpha_total)  # Sometimes alpha becomes a matrix of complex numbers.

    acc_DeKPCA[0, rep_iter] = 0
    for ii in range(pms.target_k):
        v1 = alpha_DeKPCA_j[:, ii].reshape(-1, 1)
        v2 = alpha_ggt[:, ii].reshape(-1, 1)
        acc_DeKPCA[0, rep_iter] += np.abs(v1.T @ kernel_total_j @ v2) / np.sqrt(
            v1.T @ K_mat_j[rank][rank] @ v1) / np.sqrt(v2.T @ kernel_tt @ v2)
    # -----(End) DeKPCA-----

    # -----(Start) JMLR 2010-----
    # Input: K_mat_j + N_j + pms
    # Data Preparation.
    lam = [[[] for i in range(pms.worker_num)] for j in range(pms.worker_num)]
    zeta = [[] for i in range(pms.worker_num)]

    adjust = np.zeros((pms.local_n, pms.local_n))
    alpha_total = np.zeros((pms.local_n, pms.target_k))

    tic = MPI.Wtime()
    for k_iter in range(pms.target_k):
        for nei_iter_1 in range(norm_N_j):
            index_1 = N_j[nei_iter_1, 0]
            zeta[index_1] = np.zeros((1, pms.local_n))
            for nei_iter_2 in range(norm_N_j):
                index_2 = N_j[nei_iter_2, 0]
                lam[index_1][index_2] = np.zeros((1, local_n_nei[rank][nei_iter_1]))
        tmp = (K_mat_j[rank][rank] - adjust) @ (K_mat_j[rank][rank] - adjust)
        alpha_j, first_eig = solve_global_svd(tmp, 1)

        prime_flag = 4
        RHO_BASE = 0.01 / (k_iter + 1)
        max_first_eig = comm.allreduce(first_eig, op=MPI.MAX)

        rho = np.zeros((1, pms.worker_num))
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            rho[0, nei_tmp] = RHO_BASE * math.ceil(max_first_eig)
        # Invent Initialization.
        tmp_kernel_sum = np.zeros((pms.local_n, pms.local_n))
        tmp = np.dot(K_mat_j[rank][rank], K_mat_j[rank][rank].T)
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            tmp_kernel_sum += rho[0, nei_tmp] * np.dot(K_mat_j[rank][nei_tmp], K_mat_j[rank][nei_tmp].T) \
                              + rho[0, rank] * tmp
        kernel_sum = tmp_kernel_sum - 2 * (K_mat_j[rank][rank] - adjust) @ (K_mat_j[rank][rank] - adjust)

        # Broadcast K_mat_j and rho.
        K_mat_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                K_mat_nei[i] = comm_j[i].gather(K_mat_j, root=0)
        rho_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                rho_nei[i] = comm_j[i].gather(rho, root=0)

        for ADMM_iter in range(60):
            # Broadcast \alpha and lambda.
            alpha_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    alpha_nei[i] = comm_j[i].gather(alpha_j, root=0)
            lam_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    lam_nei[i] = comm_j[i].gather(lam, root=0)
            # Update zeta.
            for nei_iter in range(norm_N_j):
                nei_tmp = N_j[nei_iter, 0]
                zeta[nei_tmp] = 0.5 * np.dot(alpha_j.T, K_mat_j[rank][rank]) \
                                + 0.5 * np.dot(alpha_nei[rank][nei_iter].T, K_mat_nei[rank][nei_iter][nei_tmp][rank]) \
                                + 0.5 * lam[rank][nei_tmp] / rho[0, nei_tmp] \
                                + 0.5 * lam_nei[rank][nei_iter][rank][nei_tmp] / rho_nei[rank][nei_iter][0, rank]
            # Broadcast zeta.
            zeta_nei = [[] for i in range(pms.worker_num)]
            for i in range(pms.worker_num):
                if rank in j_list[i]:
                    zeta_nei[i] = comm_j[i].gather(zeta, root=0)

            # Update alpha.
            tmp_sum = np.zeros((pms.local_n, 1))
            for nei_iter in range(norm_N_j):
                nei_tmp = N_j[nei_iter, 0]
                tmp_sum += np.dot(K_mat_j[rank][nei_tmp], (-lam[nei_tmp][rank] + rho[0, nei_tmp] * zeta_nei[rank][nei_iter][rank]).T) \
                           + np.dot(K_mat_j[rank][rank], (-lam[rank][nei_tmp] + rho[0, nei_tmp] * zeta[nei_tmp]).T)
            alpha_j = PGD.gradient_projection_descent(kernel_sum, -tmp_sum, alpha_j)

            prime_flag += 1
            if prime_flag >= 4:
                # Update lambda.
                for nei_iter in range(norm_N_j):
                    nei_tmp = N_j[nei_iter, 0]
                    lam[rank][nei_tmp] += rho[0, nei_tmp] * 0.5 * (np.dot(alpha_j.T, K_mat_j[rank][rank]) - zeta[nei_tmp])
                    lam[nei_tmp][rank] += rho[0, nei_tmp] * 0.5 * (np.dot(alpha_j.T, K_mat_j[rank][nei_tmp]) - zeta_nei[rank][nei_iter][rank])
                prime_flag = 0

            stop_flag_j = 0
            cnt_tmp_j = 0
            for nei_iter in range(norm_N_j):
                nei_tmp = N_j[nei_iter, 0]
                A = np.dot(K_mat_j[rank][nei_tmp].T, alpha_j)
                B = zeta_nei[rank][nei_iter][rank].T
                stop_flag_j += np.sin(subspace(A, B))
                A = np.dot(K_mat_j[rank][rank].T, alpha_j)
                B = zeta[nei_tmp].T
                stop_flag_j += np.sin(subspace(A, B))
                cnt_tmp_j += 2
            stop_flag = comm.allreduce(stop_flag_j, op=MPI.SUM)
            cnt_tmp = comm.allreduce(cnt_tmp_j, op=MPI.SUM)
            if stop_flag < (0.02 * cnt_tmp):
                print('stop iter: ', ADMM_iter, '\n.')
                break
        alpha_j = alpha_j / np.sqrt(alpha_j.T @ K_mat_j[rank][rank] @ alpha_j)
        alpha_total[:, k_iter] = alpha_j.flatten()
        adjust = K_mat_j[rank][rank] @ alpha_total @ alpha_total.T @ K_mat_j[rank][rank].T

    toc = MPI.Wtime()
    time_each = toc - tic
    time_all = comm.allreduce(time_each, op=MPI.MAX)
    if rank == 0:
        time_JMLR[0, rep_iter] = time_all
        print('JMLR running time: ', time_all, '\n')

    alpha_JMLR_j = np.real(alpha_total)

    acc_JMLR[0, rep_iter] = 0
    for ii in range(pms.target_k):
        v1 = alpha_JMLR_j[:, ii].reshape(-1, 1)
        v2 = alpha_ggt[:, ii].reshape(-1, 1)
        acc_JMLR[0, rep_iter] += np.abs(v1.T @ kernel_total_j @ v2) / np.sqrt(
            v1.T @ K_mat_j[rank][rank] @ v1) / np.sqrt(v2.T @ kernel_tt @ v2)
    # -----(End) JMLR 2010-----

mean_acc_ini = np.mean(acc_ini, axis=1)
mean_acc_DeKPCA = np.mean(acc_DeKPCA, axis=1)
mean_acc_JMLR = np.mean(acc_JMLR, axis=1)

result_acc_ini = comm.gather(mean_acc_ini, root=0)
result_acc_DeKPCA = comm.gather(mean_acc_DeKPCA, root=0)
result_acc_JMLR = comm.gather(mean_acc_JMLR, root=0)

if rank == 0:
    list_acc_ini = []
    list_acc_DeKPCA = []
    list_acc_JMLR = []
    for i in range(pms.worker_num):
        list_acc_ini.append(result_acc_ini[i][0])
        list_acc_DeKPCA.append(result_acc_DeKPCA[i][0])
        list_acc_JMLR.append(result_acc_JMLR[i][0])

if rank == 0:
    with open("expA1.1Twitter_result_time.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["Single Machine:"])
        f_csv.writerow([np.mean(time_ini, axis=1)])
        f_csv.writerow(["DeKPCA:"])
        f_csv.writerow([np.mean(time_DeKPCA, axis=1)])
        f_csv.writerow(["JMLR:"])
        f_csv.writerow([np.mean(time_JMLR, axis=1)])
    with open("expA1.1Twitter_result_acc.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["Init:"])
        f_csv.writerow([list_acc_ini])
        f_csv.writerow(["DeKPCA:"])
        f_csv.writerow([list_acc_DeKPCA])
        f_csv.writerow(["JMLR:"])
        f_csv.writerow([list_acc_JMLR])
