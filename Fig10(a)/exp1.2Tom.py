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
import scipy.io as sio
import copy


class MyPms:
    def __init__(self):
        self.worker_num = 6  # J in paper
        self.k = 50
        self.target_k = 1
        self.train_num = 200
        self.test_num = 0
        self.ill_thres = 0.01
        self.max_repeat = 1


# def generate_graph(worker_num, nei_num, type='normal'):
#     # 1.1 Calculate adjacent matrix.
#     if rank == 0:
#         if type == 'normal':
#             # adj_mat = gen_adjmat(worker_num, nei_num)
#             adj_mat = np.array([[0, 1, 0, 1, 0, 0],
#                                 [1, 0, 1, 0, 0, 0],
#                                 [0, 1, 0, 1, 0, 0],
#                                 [1, 0, 1, 0, 1, 0],
#                                 [0, 0, 0, 1, 0, 1],
#                                 [0, 0, 0, 0, 1, 0]])
#         elif type == '3':
#             adj_mat = np.array([[0, 1, 1, 0, 0],
#                                 [1, 0, 0, 0, 0],
#                                 [1, 0, 0, 1, 0],
#                                 [0, 0, 1, 0, 1],
#                                 [0, 0, 0, 1, 0]])
#         elif type == '5':
#             adj_mat = np.array([[0, 1, 0, 1, 0, 0],
#                                 [1, 0, 1, 0, 0, 0],
#                                 [0, 1, 0, 1, 0, 0],
#                                 [1, 0, 1, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1],
#                                 [0, 0, 0, 0, 1, 0]])
#
#         # print(adj_mat)
#     else:
#         adj_mat = None
#
#     adj_mat = comm.bcast(adj_mat, root=0)
#
#     # 1.2 Prepare and broadcast N_j.
#     nei = np.where(adj_mat[:, rank] == 1)
#     tmp = nei[0].reshape(-1, 1)
#     N_j = np.row_stack(([rank], tmp))  # Put j to the first.
#     norm_N_j = np.size(N_j)
#
#     # 1.3 Create new communicators based on N_j.
#     tmp = N_j.flatten()
#     a = tmp.tolist()
#     j_list = comm.allgather(a)
#     comm_j = [[] for i in range(pms.worker_num)]
#     for i in range(pms.worker_num):
#         comm_j[i] = comm.Create(comm.group.Incl(j_list[i]))
#
#     N_j_nei = [[] for i in range(pms.worker_num)]
#     for i in range(pms.worker_num):
#         if rank in j_list[i]:
#             N_j_nei[i] = comm_j[i].gather(N_j, root=0)
#
#     return adj_mat, N_j, norm_N_j, j_list, comm_j, N_j_nei


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pms = MyPms()
nei_num = 2
file_path = 'TomsHardware.mat'

group = comm.Get_group()
new_group1 = group.Incl([0, 1, 2, 3])
new_comm_1 = comm.Create(new_group1)

new_group2 = group.Incl([5])
new_comm_2 = comm.Create(new_group2)

pms.local_n = pms.train_num + pms.test_num

# 1.1 Calculate adjacent matrix.
if rank == 0:
    # adj_mat = gen_adjmat(pms.worker_num, nei_num)
    adj_mat = np.array([[0, 1, 0, 1, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0]])
    # print(adj_mat)
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

if rank == 0:
    # for i in range(result_lag.__len__()):
    #     print(i, 'lag', result_lag[i])
    with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
        f_csv = csv.writer(f)

# 1.4 Start repeating the experiment.
lam_ini = 0
lam_DeKPCA = 0
lam_DeKPCA_Ny = 0

acc_ini = np.zeros((1, pms.max_repeat))
acc_DeKPCA = np.zeros((1, pms.max_repeat))
acc_DeKPCA_Ny = np.zeros((1, pms.max_repeat))
if rank == 0:
    time_ini = np.zeros((1, pms.max_repeat))
    time_DeKPCA = np.zeros((1, pms.max_repeat))
    time_DeKPCA_Ny = np.zeros((1, pms.max_repeat))
else:
    time_ini = None
    time_DeKPCA = None
    time_DeKPCA_Ny = None

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
    X_tr_j = comm.scatter(split_X_tr, root=0)  # data: d x N
    X_tr_j = X_tr_j.T  # data: d x N
    X_tr_j = X_tr_j.astype(np.float64)
    Y_tr_j = comm.scatter(split_Y_tr, root=0)  # data: d x N
    Y_tr_j = Y_tr_j.T  # data: d x N
    Y_tr_j = Y_tr_j.astype(np.float64)
    # X_te_j = comm.bcast(X_test, root=0)  # data: d x N
    # X_te_j = X_te_j.T  # data: d x N
    # X_te_j = X_te_j.astype(np.float64)
    # Y_te_j = comm.bcast(Y_test, root=0)  # data: d x N
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
        kernel_tt = rbf_kernel(data_total.T, data_total.T, 1 / pow(pms.sigma, 2))
        kernel_tt = centralize_kernel(kernel_tt)
        alpha_ggt, _ = solve_global_svd(kernel_tt, pms.target_k)
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
            K_mat_j[index_1][index_2] = centralize_kernel(tmp_kernel)  # (K_mat_j)_{i,j} = <z(X_i), z(X_j)>

    alpha_ini, lam_ini = solve_global_svd(K_mat_j[rank][rank], pms.target_k)

    K_mat_j[rank][rank] = K_mat_j[rank][rank] + pms.ill_thres * np.min(lam_ini) / pms.local_n * np.ones(
        (np.size(K_mat_j[rank][rank], 0), np.size(K_mat_j[rank][rank], 1)))
    inv_K_mat_j = [[] for i in range(pms.worker_num)]
    for nei_iter in range(norm_N_j):
        nei_tmp = N_j[nei_iter, 0]
        inv_K_mat_j[nei_tmp] = my_inv(K_mat_j[nei_tmp][nei_tmp])
    kernel_inv = my_inv(K_mat_j[rank][rank])

    result_lag = []
    acc_ini[0, rep_iter] = 0
    for i in range(pms.target_k):
        v1 = alpha_ini[:, i].reshape(-1, 1)
        v2 = alpha_ggt[:, i].reshape(-1, 1)
        acc_ini[0, rep_iter] += np.abs(v1.T @ kernel_total_j @ v2 / np.sqrt(v1.T @ K_mat_j[rank][rank] @ v1) \
                                       / np.sqrt(v2.T @ kernel_tt @ v2))

    ttmp = np.abs(v1.T @ kernel_total_j @ v2 / np.sqrt(v1.T @ K_mat_j[rank][rank] @ v1) \
                                  / np.sqrt(v2.T @ kernel_tt @ v2))
    tmp_lag1 = comm.allgather(ttmp[0, 0])
    if rank == 0:
        result_lag.append(tmp_lag1)
        with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['init'])
            f_csv.writerow([tmp_lag1])
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

        for ADMM_iter in range(0):
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
                        tmp_z_l = H_hat * K_mat_j[nei_tmp_1][nei_tmp_2] @ inv_K_mat_j[nei_tmp_2] @ \
                                  eta_nei[rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1) \
                                  + H_hat * rho_nei[rank][nei_iter_2][0][idx_z[0][0]] * np.dot(
                            K_mat_j[nei_tmp_1][nei_tmp_2], alpha_nei[rank][nei_iter_2])
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

            alpha_j = my_inv(H_alpha) @ f_alpha
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

    v1 = alpha_j[:, 0].reshape(-1, 1)
    v2 = alpha_ggt[:, 0].reshape(-1, 1)
    acc_ = np.abs(v1.T @ kernel_total_j @ v2) / np.sqrt(v1.T @ K_mat_j[rank][rank] @ v1) \
           / np.sqrt(v2.T @ kernel_tt @ v2)
    tmp_lag = comm.allgather(acc_[0, 0])
    # Lag1 = - norm(K_mat_j[rank][rank] @ alpha_j, 2)
    # Lag2 = np.trace(eta_j.T @ K_mat_j[rank][rank] @ alpha_j @ E_j - eta_j.T @ phi_z_xi)
    # acc_ = Lag1 + Lag2
    # tmp_lag = comm.allgather(acc_)

    if rank == 0:
        result_lag.append(tmp_lag)
        with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['after1'])
            f_csv.writerow([tmp_lag])

    ####################remove 2
    if rank == 0 or rank == 1 or rank == 2 or rank == 3:
        new_rank = new_comm_1.Get_rank()
        if new_rank == 0:
            adj_mat = np.array([[0, 1, 0, 1],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [1, 0, 1, 0]])
        else:
            adj_mat = None

        adj_mat = new_comm_1.bcast(adj_mat, root=0)

        # 1.2 Prepare and broadcast N_j.
        nei = np.where(adj_mat[:, new_rank] == 1)
        tmp = nei[0].reshape(-1, 1)
        N_j = np.row_stack(([new_rank], tmp))  # Put j to the first.
        norm_N_j = np.size(N_j)

        # 1.3 Create new communicators based on N_j.
        tmp = N_j.flatten()
        a = tmp.tolist()
        j_list = new_comm_1.allgather(a)
        comm_j = [[] for i in range(pms.worker_num - 2)]
        for i in range(pms.worker_num - 2):
            comm_j[i] = new_comm_1.Create(new_comm_1.group.Incl(j_list[i]))

        N_j_nei = [[] for i in range(pms.worker_num - 2)]
        for i in range(pms.worker_num - 2):
            if new_rank in j_list[i]:
                N_j_nei[i] = comm_j[i].gather(N_j, root=0)

        local_n_nei = [[] for i in range(pms.worker_num - 2)]
        for i in range(pms.worker_num - 2):
            if new_rank in j_list[i]:
                local_n_nei[i] = comm_j[i].gather(pms.local_n, root=0)
        data_nei = [[] for i in range(pms.worker_num - 2)]
        for i in range(pms.worker_num - 2):
            if new_rank in j_list[i]:
                data_nei[i] = comm_j[i].gather(data_j, root=0)

        K_mat_j = [[[] for i in range(pms.worker_num - 2)] for j in range(pms.worker_num - 2)]
        for nei_iter_1 in range(norm_N_j):
            index_1 = N_j[nei_iter_1, 0]
            for nei_iter_2 in range(norm_N_j):
                index_2 = N_j[nei_iter_2, 0]

                data_1 = data_nei[new_rank][nei_iter_1]
                data_2 = data_nei[new_rank][nei_iter_2]
                tmp_kernel = rbf_kernel(data_1.T, data_2.T, 1 / pow(pms.sigma, 2))
                K_mat_j[index_1][index_2] = centralize_kernel(tmp_kernel)  # (K_mat_j)_{i,j} = <z(X_i), z(X_j)>

        K_mat_j[new_rank][new_rank] = K_mat_j[new_rank][new_rank] + pms.ill_thres * np.min(lam_ini) / pms.local_n * np.ones(
            (np.size(K_mat_j[new_rank][new_rank], 0), np.size(K_mat_j[new_rank][new_rank], 1)))
        inv_K_mat_j = [[] for i in range(pms.worker_num - 2)]
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            inv_K_mat_j[nei_tmp] = my_inv(K_mat_j[nei_tmp][nei_tmp])
        kernel_inv = my_inv(K_mat_j[new_rank][new_rank])

        ee_j = np.zeros((pms.worker_num - 2, 1))
        ee_j[new_rank] = 1
        ee = new_comm_1.allgather(ee_j)
        E_j = np.ones((1, norm_N_j))
        E = new_comm_1.allgather(E_j)

        xi_j = np.zeros((pms.worker_num - 2, norm_N_j))
        for i in range(norm_N_j):
            xi_j[:, i] = ee[N_j[i, 0]].flatten()
        xi = new_comm_1.allgather(xi_j)

        if rank == 3:
            eta_tmp = copy.deepcopy(eta_j)
            eta_j = eta_tmp[:, :3]

        # Increase rho with iterations.
        for ADMM_iter in range(1, 10):
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
                if nei_tmp == new_rank:
                    rho[0][nei_iter] = rho_j * first_eig
                else:
                    rho[0][nei_iter] = rho_l * first_eig
            H_hat = 1 / np.sum(rho)
            rho_nei = [[] for i in range(pms.worker_num - 2)]
            for i in range(pms.worker_num - 2):
                if new_rank in j_list[i]:
                    rho_nei[i] = comm_j[i].gather(rho, root=0)

            # Broadcast \alpha and \eta.
            alpha_nei = [[] for i in range(pms.worker_num - 2)]
            for i in range(pms.worker_num - 2):
                if new_rank in j_list[i]:
                    alpha_nei[i] = comm_j[i].gather(alpha_j, root=0)

            eta_nei = [[] for i in range(pms.worker_num - 2)]
            for i in range(pms.worker_num - 2):
                if new_rank in j_list[i]:
                    eta_nei[i] = comm_j[i].gather(eta_j, root=0)

            # DeKPCA.3.1.1 Calculate Z.
            # (*)define phi_z[l] = \phi(X_l)^T * Z * e_j (\subset R^{local_l x 1}).
            if update_flag > update_thres:
                for nei_iter_1 in range(norm_N_j):
                    nei_tmp_1 = N_j[nei_iter_1, 0]
                    phi_z[nei_tmp_1] = np.zeros((local_n_nei[new_rank][nei_iter_1], 1))
                    for nei_iter_2 in range(norm_N_j):
                        nei_tmp_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[new_rank][nei_iter_2] == new_rank)
                        # (*)define tmp_z_l = K_l^{-1} * eta_l + alpha_l * E * rho (\subset R^{local_l x |N_l|}).
                        tmp_z_l = H_hat * K_mat_j[nei_tmp_1][nei_tmp_2] @ inv_K_mat_j[nei_tmp_2] @ \
                                  eta_nei[new_rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1) \
                                  + H_hat * rho_nei[new_rank][nei_iter_2][0][idx_z[0][0]] * np.dot(
                            K_mat_j[nei_tmp_1][nei_tmp_2], alpha_nei[new_rank][nei_iter_2])
                        phi_z[nei_tmp_1] = phi_z[nei_tmp_1] + tmp_z_l

            # DeKPCA.3.1.2 Calculate ||Ze_j|| and project z_j.
            if update_flag > update_thres:
                z_norm = 1
                for nei_iter_1 in range(norm_N_j):
                    index_1 = N_j[nei_iter_1, 0]
                    idx_z = np.where(N_j_nei[new_rank][nei_iter_1] == new_rank)
                    inf_nei1 = np.dot(inv_K_mat_j[index_1], eta_nei[new_rank][nei_iter_1][:, idx_z[0][0]].reshape(-1, 1)) \
                               + rho_nei[new_rank][nei_iter_1][0][idx_z[0][0]] * alpha_nei[new_rank][nei_iter_1]
                    inf_nei1 = H_hat * inf_nei1
                    for nei_iter_2 in range(norm_N_j):
                        index_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[new_rank][nei_iter_2] == new_rank)
                        inf_nei2 = np.dot(inv_K_mat_j[index_2],
                                          eta_nei[new_rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1)) + \
                                   rho_nei[new_rank][nei_iter_2][0][idx_z[0][0]] * alpha_nei[new_rank][nei_iter_2]
                        inf_nei2 = H_hat * inf_nei2
                        z_norm += multi_dot([inf_nei1.T, K_mat_j[index_1][index_2], inf_nei2])[0][0]
                if z_norm > 1:
                    for nei_iter in range(norm_N_j):
                        nei_tmp = N_j[nei_iter, 0]
                        phi_z[nei_tmp] = phi_z[nei_tmp] / np.sqrt(z_norm)

            # DeKPCA.3.1.3 Build \phi(X_j) * Z * \xi_j
            if update_flag > update_thres:
                xx = [[] for i in range(pms.worker_num - 2)]
                for i in range(pms.worker_num - 2):
                    tmp_data = phi_z[i]
                    if new_rank in j_list[i]:
                        xx[i] = comm_j[i].gather(tmp_data, root=0)
                for i in range(norm_N_j):
                    if i == 0:
                        phi_z_xi = xx[new_rank][i]
                    else:
                        phi_z_xi = np.column_stack((phi_z_xi, xx[new_rank][i]))

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
            sum_delta_z = new_comm_1.allreduce(delta_z, op=MPI.SUM)
            if sum_delta_z < (pms.worker_num - 2) * update_thres:
                z_flag = 1
            phi_z_old = np.array(phi_z, dtype=object)

            # Compute the update flag of z.
            A = np.dot(alpha_j, E_j)
            B = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
            update_flag = np.sin(subspace(A, B))

            # DeKPCA.3.2 Update \alpha_j.
            # (*)define \alpha_j = H_alpha^-1 * f_alpha
            delta_alpha = 0
            alpha_old = np.array(alpha_j)
            H_alpha = -2 * np.dot(K_mat_j[new_rank][new_rank] - adjust, K_mat_j[new_rank][new_rank] - adjust) \
                      + multi_dot([E_j, np.diag(rho.flatten()), E_j.T])[0][0] * np.eye(pms.local_n)
            f_alpha = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten()), E_j.T]) - np.dot(eta_j, E_j.T)

            alpha_j = my_inv(H_alpha) @ f_alpha
            delta_alpha += norm(alpha_j - alpha_old)
            sum_delta_alpha = new_comm_1.allreduce(delta_alpha, op=MPI.SUM)
            if sum_delta_alpha < (pms.worker_num - 2) * update_thres:  # if alpha converges, then update eta.
                alpha_flag = 1

            # DeKPCA.3.3 Update and boardcast \eta_j.
            if z_flag == 1 and alpha_flag == 1:
                eta_old = np.array(eta_j)
                eta_j = eta_j + multi_dot([alpha_j, E_j, np.diag(rho.flatten())]) \
                        - multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
                A = np.dot(alpha_j, E_j)
                B = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
                update_flag = np.sin(subspace(A, B))
                sum_update_flag = new_comm_1.allreduce(update_flag, op=MPI.SUM)
                if sum_update_flag < (pms.worker_num - 2) * stop_flag:
                    stage += 1
                    if stage == 8:
                        if new_rank == 0:
                            print("In iter", ADMM_iter + 1, "stage reaches 6.")
                        break
                alpha_flag = 0
                z_flag = 0

            v1 = alpha_j[:, 0].reshape(-1, 1)
            v2 = alpha_ggt[:, 0].reshape(-1, 1)
            acc_ = np.abs(v1.T @ kernel_total_j @ v2) / np.sqrt(v1.T @ K_mat_j[new_rank][new_rank] @ v1) \
                   / np.sqrt(v2.T @ kernel_tt @ v2)
            tmp_lag = new_comm_1.allgather(acc_[0, 0])
            if new_rank == 0:
                result_lag.append(tmp_lag)
                with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow([tmp_lag])

    # -----(End) DeKPCA-----
    ####################remove 2
    if rank == 5:
        new_rank = new_comm_2.Get_rank()
        if new_rank == 0:
            adj_mat = np.array([[0]])
        else:
            adj_mat = None

        adj_mat = new_comm_2.bcast(adj_mat, root=0)

        # 1.2 Prepare and broadcast N_j.
        nei = np.where(adj_mat[:, new_rank] == 1)
        tmp = nei[0].reshape(-1, 1)
        N_j = np.row_stack(([new_rank], tmp))  # Put j to the first.
        norm_N_j = np.size(N_j)

        # 1.3 Create new communicators based on N_j.
        tmp = N_j.flatten()
        a = tmp.tolist()
        j_list = new_comm_2.allgather(a)
        comm_j = [[] for i in range(pms.worker_num - 5)]
        for i in range(pms.worker_num - 5):
            comm_j[i] = new_comm_2.Create(new_comm_2.group.Incl(j_list[i]))

        N_j_nei = [[] for i in range(pms.worker_num - 5)]
        for i in range(pms.worker_num - 5):
            if new_rank in j_list[i]:
                N_j_nei[i] = comm_j[i].gather(N_j, root=0)

        local_n_nei = [[] for i in range(pms.worker_num - 5)]
        for i in range(pms.worker_num - 5):
            if new_rank in j_list[i]:
                local_n_nei[i] = comm_j[i].gather(pms.local_n, root=0)
        data_nei = [[] for i in range(pms.worker_num - 5)]
        for i in range(pms.worker_num - 5):
            if new_rank in j_list[i]:
                data_nei[i] = comm_j[i].gather(data_j, root=0)

        K_mat_j = [[[] for i in range(pms.worker_num - 5)] for j in range(pms.worker_num - 5)]
        for nei_iter_1 in range(norm_N_j):
            index_1 = N_j[nei_iter_1, 0]
            for nei_iter_2 in range(norm_N_j):
                index_2 = N_j[nei_iter_2, 0]

                data_1 = data_nei[new_rank][nei_iter_1]
                data_2 = data_nei[new_rank][nei_iter_2]
                tmp_kernel = rbf_kernel(data_1.T, data_2.T, 1 / pow(pms.sigma, 2))
                K_mat_j[index_1][index_2] = centralize_kernel(tmp_kernel)  # (K_mat_j)_{i,j} = <z(X_i), z(X_j)>

        K_mat_j[new_rank][new_rank] = K_mat_j[new_rank][new_rank] + pms.ill_thres * np.min(
            lam_ini) / pms.local_n * np.ones(
            (np.size(K_mat_j[new_rank][new_rank], 0), np.size(K_mat_j[new_rank][new_rank], 1)))
        inv_K_mat_j = [[] for i in range(pms.worker_num - 5)]
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            inv_K_mat_j[nei_tmp] = my_inv(K_mat_j[nei_tmp][nei_tmp])
        kernel_inv = my_inv(K_mat_j[new_rank][new_rank])

        ee_j = np.zeros((pms.worker_num - 5, 1))
        ee_j[new_rank] = 1
        ee = new_comm_2.allgather(ee_j)
        E_j = np.ones((1, norm_N_j))
        E = new_comm_2.allgather(E_j)

        xi_j = np.zeros((pms.worker_num - 5, norm_N_j))
        for i in range(norm_N_j):
            xi_j[:, i] = ee[N_j[i, 0]].flatten()
        xi = new_comm_2.allgather(xi_j)

        if rank == 5:
            eta_tmp = copy.deepcopy(eta_j)
            eta_j = eta_tmp[:, :1]

        # Increase rho with iterations.
        for ADMM_iter in range(1, 10):
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
                if nei_tmp == new_rank:
                    rho[0][nei_iter] = rho_j * first_eig
                else:
                    rho[0][nei_iter] = rho_l * first_eig
            H_hat = 1 / np.sum(rho)
            rho_nei = [[] for i in range(pms.worker_num - 5)]
            for i in range(pms.worker_num - 5):
                if new_rank in j_list[i]:
                    rho_nei[i] = comm_j[i].gather(rho, root=0)

            # Broadcast \alpha and \eta.
            alpha_nei = [[] for i in range(pms.worker_num - 5)]
            for i in range(pms.worker_num - 5):
                if new_rank in j_list[i]:
                    alpha_nei[i] = comm_j[i].gather(alpha_j, root=0)

            eta_nei = [[] for i in range(pms.worker_num - 5)]
            for i in range(pms.worker_num - 5):
                if new_rank in j_list[i]:
                    eta_nei[i] = comm_j[i].gather(eta_j, root=0)

            # DeKPCA.3.1.1 Calculate Z.
            # (*)define phi_z[l] = \phi(X_l)^T * Z * e_j (\subset R^{local_l x 1}).
            if update_flag > update_thres:
                for nei_iter_1 in range(norm_N_j):
                    nei_tmp_1 = N_j[nei_iter_1, 0]
                    phi_z[nei_tmp_1] = np.zeros((local_n_nei[new_rank][nei_iter_1], 1))
                    for nei_iter_2 in range(norm_N_j):
                        nei_tmp_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[new_rank][nei_iter_2] == new_rank)
                        # (*)define tmp_z_l = K_l^{-1} * eta_l + alpha_l * E * rho (\subset R^{local_l x |N_l|}).
                        tmp_z_l = H_hat * K_mat_j[nei_tmp_1][nei_tmp_2] @ inv_K_mat_j[nei_tmp_2] @ \
                                  eta_nei[new_rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1) \
                                  + H_hat * rho_nei[new_rank][nei_iter_2][0][idx_z[0][0]] * np.dot(
                            K_mat_j[nei_tmp_1][nei_tmp_2], alpha_nei[new_rank][nei_iter_2])
                        phi_z[nei_tmp_1] = phi_z[nei_tmp_1] + tmp_z_l

            # DeKPCA.3.1.2 Calculate ||Ze_j|| and project z_j.
            if update_flag > update_thres:
                z_norm = 1
                for nei_iter_1 in range(norm_N_j):
                    index_1 = N_j[nei_iter_1, 0]
                    idx_z = np.where(N_j_nei[new_rank][nei_iter_1] == new_rank)
                    inf_nei1 = np.dot(inv_K_mat_j[index_1],
                                      eta_nei[new_rank][nei_iter_1][:, idx_z[0][0]].reshape(-1, 1)) \
                               + rho_nei[new_rank][nei_iter_1][0][idx_z[0][0]] * alpha_nei[new_rank][nei_iter_1]
                    inf_nei1 = H_hat * inf_nei1
                    for nei_iter_2 in range(norm_N_j):
                        index_2 = N_j[nei_iter_2, 0]
                        idx_z = np.where(N_j_nei[new_rank][nei_iter_2] == new_rank)
                        inf_nei2 = np.dot(inv_K_mat_j[index_2],
                                          eta_nei[new_rank][nei_iter_2][:, idx_z[0][0]].reshape(-1, 1)) + \
                                   rho_nei[new_rank][nei_iter_2][0][idx_z[0][0]] * alpha_nei[new_rank][nei_iter_2]
                        inf_nei2 = H_hat * inf_nei2
                        z_norm += multi_dot([inf_nei1.T, K_mat_j[index_1][index_2], inf_nei2])[0][0]
                if z_norm > 1:
                    for nei_iter in range(norm_N_j):
                        nei_tmp = N_j[nei_iter, 0]
                        phi_z[nei_tmp] = phi_z[nei_tmp] / np.sqrt(z_norm)

            # DeKPCA.3.1.3 Build \phi(X_j) * Z * \xi_j
            if update_flag > update_thres:
                xx = [[] for i in range(pms.worker_num - 5)]
                for i in range(pms.worker_num - 5):
                    tmp_data = phi_z[i]
                    if new_rank in j_list[i]:
                        xx[i] = comm_j[i].gather(tmp_data, root=0)
                for i in range(norm_N_j):
                    if i == 0:
                        phi_z_xi = xx[new_rank][i]
                    else:
                        phi_z_xi = np.column_stack((phi_z_xi, xx[new_rank][i]))

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
            sum_delta_z = new_comm_2.allreduce(delta_z, op=MPI.SUM)
            if sum_delta_z < (pms.worker_num - 5) * update_thres:
                z_flag = 1
            phi_z_old = np.array(phi_z, dtype=object)

            # Compute the update flag of z.
            A = np.dot(alpha_j, E_j)
            B = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
            update_flag = np.sin(subspace(A, B))

            # DeKPCA.3.2 Update \alpha_j.
            # (*)define \alpha_j = H_alpha^-1 * f_alpha
            delta_alpha = 0
            alpha_old = np.array(alpha_j)
            H_alpha = -2 * np.dot(K_mat_j[new_rank][new_rank] - adjust, K_mat_j[new_rank][new_rank] - adjust) \
                      + multi_dot([E_j, np.diag(rho.flatten()), E_j.T])[0][0] * np.eye(pms.local_n)
            f_alpha = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten()), E_j.T]) - np.dot(eta_j,
                                                                                                           E_j.T)

            alpha_j = my_inv(H_alpha) @ f_alpha
            delta_alpha += norm(alpha_j - alpha_old)
            sum_delta_alpha = new_comm_2.allreduce(delta_alpha, op=MPI.SUM)
            if sum_delta_alpha < (pms.worker_num - 5) * update_thres:  # if alpha converges, then update eta.
                alpha_flag = 1

            # DeKPCA.3.3 Update and boardcast \eta_j.
            if z_flag == 1 and alpha_flag == 1:
                eta_old = np.array(eta_j)
                eta_j = eta_j + multi_dot([alpha_j, E_j, np.diag(rho.flatten())]) \
                        - multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
                A = np.dot(alpha_j, E_j)
                B = multi_dot([inv_K_mat_j[new_rank], phi_z_xi, np.diag(rho.flatten())])
                update_flag = np.sin(subspace(A, B))
                sum_update_flag = new_comm_2.allreduce(update_flag, op=MPI.SUM)
                if sum_update_flag < (pms.worker_num - 5) * stop_flag:
                    stage += 1
                    if stage == 8:
                        if new_rank == 0:
                            print("In iter", ADMM_iter + 1, "stage reaches 6.")
                        break
                alpha_flag = 0
                z_flag = 0

            v1 = alpha_j[:, 0].reshape(-1, 1)
            v2 = alpha_ggt[:, 0].reshape(-1, 1)
            acc_ = np.abs(v1.T @ kernel_total_j @ v2) / np.sqrt(v1.T @ K_mat_j[new_rank][new_rank] @ v1) \
                   / np.sqrt(v2.T @ kernel_tt @ v2)
            tmp_lag = new_comm_2.allgather(acc_[0, 0])
            if new_rank == 0:
                result_lag.append(tmp_lag)
                with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow([tmp_lag])

    # -----(End) DeKPCA-----

if rank == 0:
    with open("exp1.2Tom_result_acc.csv", "a", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['\n'])

