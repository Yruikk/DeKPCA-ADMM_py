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
        self.worker_num = 20  # J in paper
        self.k = 50
        self.target_k = 1
        self.test_num = 0
        self.ill_thres = 0.01
        self.max_repeat = 100


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##### N_local = 200
pms = MyPms()
pms.train_num = 200

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
lam_DeKPCA = 0

acc_DeKPCA = np.zeros((1, pms.max_repeat))
if rank == 0:
    time_DeKPCA = np.zeros((1, pms.max_repeat))
else:
    time_DeKPCA = None

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

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        X_train = min_max_scaler.fit_transform(X_tr)

        split_X_tr = np.array_split(X_train, pms.worker_num, axis=0)
    else:
        split_X_tr = None
    X_tr_j = comm.scatter(split_X_tr, root=0)  # data: N x d
    X_tr_j = X_tr_j.T  # data: d x N
    X_tr_j = X_tr_j.astype(np.float64)
    data_j = np.array(X_tr_j)

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
    else:
        data_total = None
    data_total = comm.bcast(data_total, root=0)
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
                rho_l = rho_j
            elif stage == 2:
                rho_l = rho_j
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
    # -----(End) DeKPCA-----

mean_acc_DeKPCA = np.mean(acc_DeKPCA, axis=1)

result_acc_DeKPCA = comm.gather(mean_acc_DeKPCA, root=0)

if rank == 0:
    list_acc_DeKPCA = []
    for i in range(pms.worker_num):
        list_acc_DeKPCA.append(result_acc_DeKPCA[i][0])

if rank == 0:
    with open("exp_Table_time_200.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["DeKPCA:"])
        f_csv.writerow([np.mean(time_DeKPCA, axis=1)])


##### N_local = 1000
pms = MyPms()
pms.train_num = 1000

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
lam_DeKPCA = 0

acc_DeKPCA = np.zeros((1, pms.max_repeat))
if rank == 0:
    time_DeKPCA = np.zeros((1, pms.max_repeat))
else:
    time_DeKPCA = None

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

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        X_train = min_max_scaler.fit_transform(X_tr)

        split_X_tr = np.array_split(X_train, pms.worker_num, axis=0)
    else:
        split_X_tr = None
    X_tr_j = comm.scatter(split_X_tr, root=0)  # data: N x d
    X_tr_j = X_tr_j.T  # data: d x N
    X_tr_j = X_tr_j.astype(np.float64)
    data_j = np.array(X_tr_j)

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
    else:
        data_total = None
    data_total = comm.bcast(data_total, root=0)
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
                rho_l = rho_j
            elif stage == 2:
                rho_l = rho_j
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
    # -----(End) DeKPCA-----

mean_acc_DeKPCA = np.mean(acc_DeKPCA, axis=1)

result_acc_DeKPCA = comm.gather(mean_acc_DeKPCA, root=0)

if rank == 0:
    list_acc_DeKPCA = []
    for i in range(pms.worker_num):
        list_acc_DeKPCA.append(result_acc_DeKPCA[i][0])

if rank == 0:
    with open("exp_Table_time_1000.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["DeKPCA:"])
        f_csv.writerow([np.mean(time_DeKPCA, axis=1)])


##### N_local = 2000
pms = MyPms()
pms.train_num = 2000

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
lam_DeKPCA = 0

acc_DeKPCA = np.zeros((1, pms.max_repeat))
if rank == 0:
    time_DeKPCA = np.zeros((1, pms.max_repeat))
else:
    time_DeKPCA = None

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

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        X_train = min_max_scaler.fit_transform(X_tr)

        split_X_tr = np.array_split(X_train, pms.worker_num, axis=0)
    else:
        split_X_tr = None
    X_tr_j = comm.scatter(split_X_tr, root=0)  # data: N x d
    X_tr_j = X_tr_j.T  # data: d x N
    X_tr_j = X_tr_j.astype(np.float64)
    data_j = np.array(X_tr_j)

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
    else:
        data_total = None
    data_total = comm.bcast(data_total, root=0)
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
                rho_l = rho_j
            elif stage == 2:
                rho_l = rho_j
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
    # -----(End) DeKPCA-----

mean_acc_DeKPCA = np.mean(acc_DeKPCA, axis=1)

result_acc_DeKPCA = comm.gather(mean_acc_DeKPCA, root=0)

if rank == 0:
    list_acc_DeKPCA = []
    for i in range(pms.worker_num):
        list_acc_DeKPCA.append(result_acc_DeKPCA[i][0])

if rank == 0:
    with open("exp_Table_time_2000.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["DeKPCA:"])
        f_csv.writerow([np.mean(time_DeKPCA, axis=1)])
