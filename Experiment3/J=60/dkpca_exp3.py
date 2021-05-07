import mpi4py.MPI as MPI
import numpy as np
from func import centralize_kernel, solve_global_svd, my_inv, subspace, gen_adjmat
from scipy.linalg import orth, norm
from numpy.linalg import multi_dot, eig
import csv
import time
from scipy.io import loadmat
from sklearn.metrics.pairwise import rbf_kernel


class MyPms:
    def __init__(self):
        self.worker_num = 60  # J in paper
        self.m = 500  # M in paper
        self.local_n = 100
        self.target_k = 1
        self.sigma = np.sqrt(self.m) / 0.01
        self.ill_thres = 0.01
        self.max_repeat = 100


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pms = MyPms()
nei_num = 4

# 1.1 Calculate adjacent matrix. (adj_mat[j][j] = 0 for now)
if rank == 0:
    adj_mat = gen_adjmat(pms.worker_num, nei_num)
else:
    adj_mat = None

adj_mat = comm.bcast(adj_mat, root=0)

# 1.2 Prepare and broadcast N_j and auxiliary matrices(e_j, E_j).
nei = np.where(adj_mat[:, rank] == 1)
tmp = nei[0].reshape(-1, 1)
N_j = np.row_stack(([rank], tmp))  # Put j to the first.
norm_N_j = np.size(N_j)

ee_j = np.zeros((pms.worker_num, 1))
ee_j[rank] = 1
ee = comm.allgather(ee_j)
E_j = np.ones((1, norm_N_j))
E = comm.allgather(E_j)

# 1.3 Prepare \xi_j based on N_j.
xi_j = np.zeros((pms.worker_num, norm_N_j))
for i in range(norm_N_j):
    xi_j[:, i] = ee[N_j[i, 0]].flatten()
xi = comm.allgather(xi_j)

# 1.4 Create new communicators based on N_j.
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

result_algo = np.zeros((pms.max_repeat, 1))
result_algo_j = np.zeros((pms.max_repeat, 1))
time_algo = np.zeros((pms.max_repeat, 1))
time_cen = np.zeros((pms.max_repeat, 1))
time_cen_comm = np.zeros((pms.max_repeat, 1))
time_cen_cal = np.zeros((pms.max_repeat, 1))
for out_iter in range(pms.max_repeat):
    # -----data_generation_start-----
    if rank == 0:
        m_data = loadmat("mnist_all.mat")
        data0 = m_data["train0"]
        data3 = m_data["train3"]
        data5 = m_data["train5"]
        data8 = m_data["train8"]
        np.random.shuffle(data0)
        np.random.shuffle(data3)
        np.random.shuffle(data5)
        np.random.shuffle(data8)
        data0 = data0[0:int(pms.worker_num * pms.local_n / 4), :]
        data3 = data3[0:int(pms.worker_num * pms.local_n / 4), :]
        data5 = data5[0:int(pms.worker_num * pms.local_n / 4), :]
        data8 = data8[0:int(pms.worker_num * pms.local_n / 4), :]
        split_data0 = np.array_split(data0, pms.worker_num, axis=0)
        split_data3 = np.array_split(data3, pms.worker_num, axis=0)
        split_data5 = np.array_split(data5, pms.worker_num, axis=0)
        split_data8 = np.array_split(data8, pms.worker_num, axis=0)
    else:
        split_data0 = None
        split_data3 = None
        split_data5 = None
        split_data8 = None
    data_j0 = comm.scatter(split_data0, root=0)
    data_j3 = comm.scatter(split_data3, root=0)
    data_j5 = comm.scatter(split_data5, root=0)
    data_j8 = comm.scatter(split_data8, root=0)
    data_j = np.row_stack((data_j0, data_j3, data_j5, data_j8))
    np.random.shuffle(data_j)
    data_j = data_j.T
    data_j = data_j.astype(np.float64)

    local_n_nei = [[] for i in range(pms.worker_num)]
    for i in range(pms.worker_num):
        if rank in j_list[i]:
            local_n_nei[i] = comm_j[i].gather(pms.local_n, root=0)
    data_nei = [[] for i in range(pms.worker_num)]
    for i in range(pms.worker_num):
        if rank in j_list[i]:
            data_nei[i] = comm_j[i].gather(data_j, root=0)
    # -----data_generation_end-----

    s = MPI.Wtime()

    # -----distributed KPCA-----
    # 2.1 Calculate and boardcast kernel matrices.
    # (*)In variable naming, we use K instead of kernel for brevity.
    K_mat_j = [[[] for i in range(pms.worker_num)] for j in range(pms.worker_num)]
    for nei_iter_1 in range(norm_N_j):
        index_1 = N_j[nei_iter_1, 0]
        for nei_iter_2 in range(norm_N_j):
            index_2 = N_j[nei_iter_2, 0]

            data_1 = data_nei[rank][nei_iter_1]
            data_2 = data_nei[rank][nei_iter_2]
            tmp_kernel = rbf_kernel(data_1.T, data_2.T, 1 / pow(pms.sigma, 2))
            K_mat_j[index_1][index_2] = centralize_kernel(tmp_kernel)
    alpha_ini, ss = solve_global_svd(K_mat_j[rank][rank], pms.target_k)
    K_mat_j[rank][rank] = K_mat_j[rank][rank] + pms.ill_thres * ss / pms.local_n \
                          * np.ones((np.size(K_mat_j[rank][rank], 0), np.size(K_mat_j[rank][rank], 1)))
    inv_K_mat_j = [[] for i in range(pms.worker_num)]
    for nei_iter in range(norm_N_j):
        nei_tmp = N_j[nei_iter, 0]
        inv_K_mat_j[nei_tmp] = my_inv(K_mat_j[nei_tmp][nei_tmp])

    # 2.2 Initialize \alpha_j with local data. Initialize \eta_j = 0.
    alpha_j = np.array(alpha_ini)
    eta_j = np.zeros((pms.local_n, norm_N_j))

    # 3. ADMM START
    stage = 1
    update_flag = 1
    update_thres = 1e-3
    stop_flag = 1e-4
    phi_z = [[] for i in range(pms.worker_num)]
    phi_z_old = [[] for i in range(pms.worker_num)]
    for nei_iter in range(norm_N_j):
        nei_tmp = N_j[nei_iter, 0]
        phi_z_old[nei_tmp] = np.zeros((np.size(data_nei[rank][nei_iter], 1), 1))

    for ADMM_iter in range(20):
        # increase rho with the iteration
        rho_j = 100
        if stage == 1:
            rho_l = 1
        elif stage == 2:
            rho_l = 50
        else:
            rho_l = 100
        rho = np.zeros((1, norm_N_j))
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            if nei_tmp == rank:
                rho[0][nei_iter] = rho_j * local_n_nei[rank][nei_iter]
            else:
                rho[0][nei_iter] = rho_l * local_n_nei[rank][nei_iter]
        H_hat = 1 / np.sum(rho)
        rho_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                rho_nei[i] = comm_j[i].gather(rho, root=0)

        alpha_flag = 0
        z_flag = 0

        # Broadcast \alpha and \eta.
        alpha_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                alpha_nei[i] = comm_j[i].gather(alpha_j, root=0)
        eta_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                eta_nei[i] = comm_j[i].gather(eta_j, root=0)

        # 3.1.1 Calculate Z.
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

        # 3.1.2 Calculate ||Ze_j|| and project z_j.
        if update_flag > update_thres:
            z_norm = 0
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

        # 3.1.3 Build \phi(X_j) * Z * \xi_j
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

        # 3.1.4 Align phi_z.
        if update_flag > update_thres:
            for nei_iter in range(norm_N_j):
                if nei_iter != 0:
                    direction = np.dot(phi_z_xi[:, nei_iter].T, phi_z_xi[:, 0])
                    if direction < 0:
                        phi_z_xi[:, nei_iter] = -phi_z_xi[:, nei_iter]

        # 3.1.5 Compute the convergence of z.
        delta_z = 0
        for nei_iter in range(norm_N_j):
            nei_tmp = N_j[nei_iter, 0]
            delta_z += norm(phi_z[nei_tmp] - phi_z_old[nei_tmp])
        sum_delta_z = comm.allreduce(delta_z, op=MPI.SUM)
        if sum_delta_z < pms.worker_num * update_thres:
            z_flag = 1
        phi_z_old = np.array(phi_z, dtype=object)

        # 3.2 Calculate \alpha_j.
        # (*)define \alpha_j = H_alpha^-1 * f_alpha
        delta_alpha = 0
        alpha_old = np.array(alpha_j)
        H_alpha = -2 * np.dot(K_mat_j[rank][rank], K_mat_j[rank][rank]) \
                  + multi_dot([E_j, np.diag(rho.flatten()), E_j.T])[0][0] * np.eye(pms.local_n)

        f_alpha = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten()), E_j.T]) - np.dot(eta_j, E_j.T)

        e_vals, e_vecs = eig(H_alpha)
        inv_e_vals = np.where(e_vals > 1e-4, 1 / e_vals, e_vals)
        inv_Sigma = np.diag(inv_e_vals)
        alpha_j = multi_dot([e_vecs, inv_Sigma, np.dot(e_vecs.T, f_alpha)])

        delta_alpha += norm(alpha_j - alpha_old)
        sum_delta_alpha = comm.allreduce(delta_alpha, op=MPI.SUM)
        if sum_delta_alpha < pms.worker_num * update_thres:  # if alpha converges, then update eta.
            alpha_flag = 1

        # 3.3 Update and boardcast \eta_j.
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
        A = np.dot(alpha_j, E_j)
        B = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten())])
        update_flag = np.sin(subspace(A, B))

    e = MPI.Wtime()
    time_each = e - s
    time_all = comm.allreduce(time_each, op=MPI.MAX)
    time_algo[out_iter][0] = time_all

    # -----compute and compare the similarity-----
    # Centralized DKPCA START.
    if rank == 0:
        s_comm = MPI.Wtime()
    data = comm.gather(data_j, root=0)  # 数据放在了一起
    if rank == 0:
        all_data = data[0]
        for i in range(pms.worker_num - 1):
            all_data = np.column_stack((all_data, data[i + 1]))
        time_tmp = MPI.Wtime()
        K_X_X = rbf_kernel(all_data.T, all_data.T, 1 / pow(pms.sigma, 2))
        K_X_X = centralize_kernel(K_X_X)
        alpha_ggt, _ = solve_global_svd(K_X_X, pms.target_k)
        e_cal = MPI.Wtime()
        time_cen_comm[out_iter][0] = time_tmp - s_comm
        time_cen_cal[out_iter][0] = e_cal - time_tmp
        time_cen[out_iter][0] = e_cal - s_comm
    else:
        all_data = None
        alpha_ggt = None
        K_X_X = None

    # Centralized DKPCA END.
    all_data = comm.bcast(all_data, root=0)
    alpha_ggt = comm.bcast(alpha_ggt, root=0)
    K_X_X = comm.bcast(K_X_X, root=0)

    # Calculate similarity between alpha_j/ini/Nj and alpha_ggt (global ground-truth).
    K_Xj_X = rbf_kernel(data_j.T, all_data.T, 1 / pow(pms.sigma, 2))
    K_Xj_X = centralize_kernel(K_Xj_X)

    K_Xj_Xnei = np.array(K_mat_j[rank][rank])
    for nei_iter in range(norm_N_j - 1):
        nei_tmp = N_j[nei_iter + 1, 0]
        K_Xj_Xnei = np.column_stack((K_Xj_Xnei, K_mat_j[rank][nei_tmp]))

    alpha_N_j, _ = solve_global_svd(np.dot(K_Xj_Xnei, K_Xj_Xnei.T), pms.target_k)

    quality_algo_j = multi_dot([alpha_j.T, K_Xj_X, alpha_ggt]) \
                     / np.sqrt(multi_dot([alpha_j.T, K_mat_j[rank][rank], alpha_j])[0][0]) \
                     / np.sqrt(multi_dot([alpha_ggt.T, K_X_X, alpha_ggt])[0][0])
    quality_algo_j = np.abs(quality_algo_j)
    result_algo_j[out_iter][0] = quality_algo_j

title = "algo result in rank" + str(int(rank)) + " when nei=" + str(int(nei_num)) + ".txt"
np.savetxt(title, result_algo_j)
final_result = comm.gather(result_algo_j, root=0)
mean_result = comm.allreduce(result_algo_j, op=MPI.SUM)
mean_result = np.mean(mean_result) / pms.worker_num
if rank == 0:
    title = "time_cen.txt"
    np.savetxt(title, time_cen)
    title = "time_algo.txt"
    np.savetxt(title, time_algo)
    with open("exp3_result_60.csv", "w", newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["algo from 0~J:"])
        f_csv.writerow([mean_result])
        f_csv.writerow(["time_algo:"])
        f_csv.writerow([np.mean(time_algo[:, 0])])
        f_csv.writerow(["time_cen:"])
        f_csv.writerow([np.mean(time_cen[:, 0])])
        f_csv.writerow(["time_cen_comm:"])
        f_csv.writerow([np.mean(time_cen_comm[:, 0])])
        f_csv.writerow(["time_cen_cal:"])
        f_csv.writerow([np.mean(time_cen_cal[:, 0])])