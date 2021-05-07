import mpi4py.MPI as MPI
import numpy as np
from func import centralize_kernel, solve_global_svd, my_inv, subspace, gen_adjmat
from scipy.linalg import orth, norm
from numpy.linalg import multi_dot, eig
import csv
from scipy.io import loadmat
from sklearn.metrics.pairwise import rbf_kernel
import time


class MyPms:
    def __init__(self):
        self.worker_num = 20  # J in paper
        self.m = 500  # M in paper
        self.local_n = 100
        self.target_k = 1
        self.sigma = np.sqrt(self.m) / 0.01
        self.ill_thres = 0.01
        self.max_repeat = 100
        self.iterations = 10


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pms = MyPms()
nei_list = [2, 4, 6, 8, 10, 12]

result_summary = [np.zeros((pms.max_repeat, pms.iterations)) for j in range(len(nei_list))]
result_summary_j = [np.zeros((pms.max_repeat, pms.iterations)) for j in range(len(nei_list))]
result_ini = np.zeros((pms.max_repeat, len(nei_list)))
result_ini_j = np.zeros((pms.max_repeat, len(nei_list)))
result_nei = np.zeros((pms.max_repeat, len(nei_list)))
result_nei_j = np.zeros((pms.max_repeat, len(nei_list)))
final_result = np.zeros((len(nei_list), pms.iterations))
time_algo = np.zeros((pms.max_repeat, len(nei_list)))
time_cen = np.zeros((pms.max_repeat, len(nei_list)))
time_cen_comm = np.zeros((pms.max_repeat, len(nei_list)))
time_cen_cal = np.zeros((pms.max_repeat, len(nei_list)))

data = loadmat("mnist_all.mat")
mat_data0 = data["train0"]
mat_data8 = data["train8"]
for0 = loadmat("for0.mat")
for8 = loadmat("for8.mat")
index0 = for0["for0"]
index8 = for8["for8"]
for_suffle = loadmat("forshuffle.mat")
shuffle = for_suffle["forshuffle"]
for out_iter in range(pms.max_repeat):
    time_cen_flag = True
    # -----data_generation_start-----
    if rank == 0:
        data0 = mat_data0[index0[out_iter], :]
        data8 = mat_data8[index8[out_iter], :]
        split_data0 = np.array_split(data0, pms.worker_num, axis=0)
        split_data8 = np.array_split(data8, pms.worker_num, axis=0)
    else:
        split_data0 = None
        split_data8 = None
    data_j0 = comm.scatter(split_data0, root=0)
    data_j8 = comm.scatter(split_data8, root=0)
    tmp_data = np.row_stack((data_j0, data_j8))
    data_j = tmp_data[shuffle[out_iter], :]
    data_j = data_j.T
    data_j = data_j.astype(np.float64)
    # print(rank, data_j.shape[0], data_j.shape[1])

    for nei_index in range(len(nei_list)):
        nei_num = nei_list[nei_index]

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

        local_n_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                local_n_nei[i] = comm_j[i].gather(pms.local_n, root=0)
        data_nei = [[] for i in range(pms.worker_num)]
        for i in range(pms.worker_num):
            if rank in j_list[i]:
                data_nei[i] = comm_j[i].gather(data_j, root=0)
        # -----data_distribute_end-----
        time_each = 0
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

        tmp_time = MPI.Wtime()
        time_each = time_each + (tmp_time - s)

        for ADMM_iter in range(pms.iterations):
            ADMM_s = MPI.Wtime()
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
                            K_mat_j[nei_tmp_1][nei_tmp_2], alpha_nei[rank][nei_iter_2])
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
                        for i in range(pms.iterations - ADMM_iter):
                            result_summary[nei_index][out_iter][ADMM_iter + i] = result_summary[nei_index][out_iter][
                                ADMM_iter - 1]
                        if rank == 0:
                            print("In iter", ADMM_iter + 1, "stage reaches 4.")
                        break
            A = np.dot(alpha_j, E_j)
            B = multi_dot([inv_K_mat_j[rank], phi_z_xi, np.diag(rho.flatten())])
            update_flag = np.sin(subspace(A, B))

            ADMM_e = MPI.Wtime()
            time_each = time_each + (ADMM_e - ADMM_s)

            # -----compute and compare the similarity-----
            if ADMM_iter == 0:
                # Centralized DKPCA START.
                if rank == 0:
                    s_comm = MPI.Wtime()
                data = comm.gather(data_j, root=0)
                if rank == 0:
                    all_data = data[0]
                    for i in range(pms.worker_num - 1):
                        all_data = np.column_stack((all_data, data[i + 1]))
                    time_tmp = MPI.Wtime()
                    K_X_X = rbf_kernel(all_data.T, all_data.T, 1 / pow(pms.sigma, 2))
                    K_X_X = centralize_kernel(K_X_X)
                    alpha_ggt, _ = solve_global_svd(K_X_X, pms.target_k)
                    e_cal = MPI.Wtime()
                    time_cen_comm[out_iter][nei_index] = time_tmp - s_comm
                    time_cen_cal[out_iter][nei_index] = e_cal - time_tmp
                    time_cen[out_iter][nei_index] = e_cal - s_comm
                else:
                    all_data = None
                    alpha_ggt = None
                    K_X_X = None

                # Centralized DKPCA END.
                all_data = comm.bcast(all_data, root=0)
                alpha_ggt = comm.bcast(alpha_ggt, root=0)
                K_X_X = comm.bcast(K_X_X, root=0)

                alpha_ggt, _ = solve_global_svd(K_X_X, pms.target_k)
                # Centralized DKPCA END.

                # Calculate similarity between alpha_j/ini/Nj and alpha_ggt (global ground-truth).
                K_Xj_X = rbf_kernel(data_j.T, all_data.T, 1 / pow(pms.sigma, 2))
                K_Xj_X = centralize_kernel(K_Xj_X)

                K_Xj_Xnei = np.array(K_mat_j[rank][rank])
                for nei_iter in range(norm_N_j - 1):
                    nei_tmp = N_j[nei_iter + 1, 0]
                    K_Xj_Xnei = np.column_stack((K_Xj_Xnei, K_mat_j[rank][nei_tmp]))

                alpha_N_j, _ = solve_global_svd(np.dot(K_Xj_Xnei, K_Xj_Xnei.T), pms.target_k)

                quality_ini_j = multi_dot([alpha_ini.T, K_Xj_X, alpha_ggt]) \
                                / np.sqrt(multi_dot([alpha_ini.T, K_mat_j[rank][rank], alpha_ini])[0][0]) \
                                / np.sqrt(multi_dot([alpha_ggt.T, K_X_X, alpha_ggt])[0][0])
                quality_ini_j = np.abs(quality_ini_j)
                result_ini_j[out_iter][nei_index] = quality_ini_j
                quality_local = comm.allreduce(quality_ini_j, op=MPI.SUM)
                quality_local = quality_local / pms.worker_num
                result_ini[out_iter][nei_index] = quality_local

                quality_N_j = multi_dot([alpha_N_j.T, K_Xj_X, alpha_ggt]) \
                              / np.sqrt(multi_dot([alpha_N_j.T, K_mat_j[rank][rank], alpha_N_j])[0][0]) \
                              / np.sqrt(multi_dot([alpha_ggt.T, K_X_X, alpha_ggt])[0][0])
                quality_N_j = np.abs(quality_N_j)
                result_nei_j[out_iter][nei_index] = quality_N_j
                quality_nei = comm.allreduce(quality_N_j, op=MPI.SUM)
                quality_nei = quality_nei / pms.worker_num
                result_nei[out_iter][nei_index] = quality_nei

            quality_algo_j = multi_dot([alpha_j.T, K_Xj_X, alpha_ggt]) \
                             / np.sqrt(multi_dot([alpha_j.T, K_mat_j[rank][rank], alpha_j])[0][0]) \
                             / np.sqrt(multi_dot([alpha_ggt.T, K_X_X, alpha_ggt])[0][0])
            quality_algo_j = np.abs(quality_algo_j)
            result_summary_j[nei_index][out_iter][ADMM_iter] = quality_algo_j
            quality_algo = comm.allreduce(quality_algo_j, op=MPI.SUM)
            quality_algo = quality_algo / pms.worker_num
            result_summary[nei_index][out_iter][ADMM_iter] = quality_algo

        time_all = comm.allreduce(time_each, op=MPI.MAX)
        time_algo[out_iter][nei_index] = time_all

title_ini = "ini result in rank" + str(int(rank)) + ".txt"
np.savetxt(title_ini, result_ini_j)
title_nei = "nei result in rank" + str(int(rank)) + ".txt"
np.savetxt(title_nei, result_nei_j)
for nei_index in range(len(nei_list)):
    tmp_idx = np.where(result_ini[:, nei_index] < 0.7)
    result_summary[nei_index] = np.delete(result_summary[nei_index], tmp_idx[0], axis=0)
    tmp_ini = np.delete(result_ini[:, nei_index], tmp_idx[0], axis=0)
    tmp_nei = np.delete(result_nei[:, nei_index], tmp_idx[0], axis=0)
    if rank == 0:
        with open("exp1_result.csv", "a", newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["ini", "nei"])
            f_csv.writerow([np.mean(tmp_ini), np.mean(tmp_nei)])
            f_csv.writerow(["time_algo:"])
            f_csv.writerow([np.mean(time_algo[:, nei_index])])
            f_csv.writerow(["time_cen:"])
            f_csv.writerow([np.mean(time_cen[:, nei_index])])
            f_csv.writerow(["time_cen_comm:"])
            f_csv.writerow([np.mean(time_cen_comm[:, nei_index])])
            f_csv.writerow(["time_cen_cal:"])
            f_csv.writerow([np.mean(time_cen_cal[:, nei_index])])

    nei_num = nei_list[nei_index]
    title = "algo result in rank" + str(int(rank)) + " when nei=" + str(int(nei_num)) + ".txt"
    np.savetxt(title, result_summary_j[nei_index])

    for i in range(pms.iterations):
        final_result[nei_index][i] = np.mean(result_summary[nei_index][:, i])
    if rank == 0:
        with open("exp1_result.csv", "a", newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
            f_csv.writerow([final_result[nei_index][0], final_result[nei_index][1], final_result[nei_index][2],
                            final_result[nei_index][3], final_result[nei_index][4], final_result[nei_index][5],
                            final_result[nei_index][6], final_result[nei_index][7], final_result[nei_index][8],
                            final_result[nei_index][9]])
