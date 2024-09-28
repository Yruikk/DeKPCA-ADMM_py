import numpy as np
from scipy.linalg import orth, norm, svd
from numpy.linalg import inv, multi_dot, eig, cond


def find_digraph(undigraph):
    n = undigraph.shape[0]
    tmp = np.sum(undigraph, axis=0)
    digraph = np.identity(n)

    idx = np.argmin(tmp)
    record = []
    start = idx
    for i in range(n):
        record.append(idx)
        min_d = float("inf")
        min_idx = 0
        for iter in range(n):
            if (iter != idx) and (iter != start) and (undigraph[idx][iter] == 1) and (
                    np.sum(digraph[:, iter]) == 1) and (np.sum(digraph[iter, :]) == 1) and (tmp[iter] < min_d):
                min_d = tmp[iter]
                min_idx = iter
        if min_idx:
            digraph[idx][min_idx] = 1
            idx = min_idx
        else:
            if not (undigraph[idx][start]):
                print("fail\n")
                break
            else:
                digraph[idx][start] = 1
                record.append(start)
                # print("success\n")
    # print(record)
    return digraph


def Gnoisegen(sig_in, SNR):
    sig_in = sig_in.flatten()
    len_signal = len(sig_in)
    noise = np.random.randn(len_signal)
    signal_power = np.sum(sig_in * sig_in) / len_signal
    noise_power = np.sum(noise * noise) / len_signal
    noise_var = signal_power / (np.power(10., (SNR / 10)))
    noise = np.sqrt(noise_var / noise_power) * noise
    sig_out = sig_in + noise
    sig_out = sig_out.reshape(-1, 1)
    snr = np.sum(sig_in * sig_in) / np.sum(noise * noise)
    return sig_out, snr


def gen_data(U_gt, U_gt2, local_n, m):
    thres = 80
    beta = 10
    a = np.row_stack((thres, thres / 2, thres / 4, beta * np.random.randn(m - 3, 1)))
    b = np.array(sorted(a, reverse=True)).flatten()
    data_j = np.zeros((m, local_n))
    sigma_gt = np.diag(b)
    for i in range(local_n):
        tmp = np.random.rand(m, 1)
        tmp = tmp / norm(tmp)
        data_j[:, i] = multi_dot([U_gt + U_gt2, sigma_gt, tmp]).flatten()

    return data_j, local_n


def gen_data_noise(U_gt, U_gt2, local_n, m, worker_num, snr):
    thres = 80
    beta = 10
    a = np.row_stack((thres, thres / 2, thres / 4, beta * np.random.randn(m - 3, 1)))
    b = np.array(sorted(a, reverse=True)).flatten()
    data_j = np.zeros((m, local_n))
    sigma_gt = np.diag(b)
    for i in range(local_n):
        tmp = np.random.rand(m, 1)
        tmp = tmp / norm(tmp)
        data_j[:, i] = multi_dot([U_gt + U_gt2, sigma_gt, tmp]).flatten()

    noise_data = np.zeros((data_j.shape[0], data_j.shape[1]))
    for i in range(local_n):
        noise_data[:, i] = Gnoisegen(data_j[:, i], snr).flatten()

    return data_j, noise_data, local_n


def cal_RBF_fun(x, y, sigma):
    val = np.exp(-pow(norm(x - y, 2), 2) / pow(sigma, 2))
    return val


def cal_RBF(X, Y, sigma):
    kernel = np.zeros((np.size(X, 1), np.size(Y, 1)))
    for iter_i in range(np.size(X, 1)):
        for iter_j in range(np.size(Y, 1)):
            kernel[iter_i, iter_j] = cal_RBF_fun(X[:, iter_i], Y[:, iter_j], sigma)

    return kernel


def centralize_kernel(kernel):
    m = np.size(kernel, 0)
    n = np.size(kernel, 1)
    mean_m = np.ones((m, m)) / m
    mean_n = np.ones((n, n)) / n
    nor_kernel = kernel - np.dot(kernel, mean_n) - np.dot(mean_m, kernel) + multi_dot([mean_m, kernel, mean_n])

    return nor_kernel


def solve_global_svd(cor_mat_noise, target_k):
    u, s, v = svd(cor_mat_noise)
    v = v.T
    ss = s[0:target_k]
    s = np.diag(s)
    Tr = np.trace(s[0:target_k, 0:target_k])
    W = v[:, 0:target_k]
    return W, ss


def my_inv(mat):
    threshold = 1e-3
    e_vals, e_vecs = eig(mat)
    e_vals = np.real(e_vals)
    e_vecs = np.real(e_vecs)
    inv_e_vals = np.where(e_vals > threshold, 1 / e_vals, e_vals)
    inv_Sigma = np.diag(inv_e_vals)
    inv_mat = multi_dot([e_vecs, inv_Sigma, e_vecs.T])

    return inv_mat


def swap(A, B):
    return B, A


def subspace(A, B):
    A = orth(A)
    B = orth(B)
    if np.size(A, 1) < np.size(B, 1):
        A, B = swap(A, B)

    B = B - np.dot(A, np.dot(A.T, B))
    theta = np.arcsin(min(1, norm(B)))
    return theta


def gen_adjmat(worker_num, nei):  # nei_num doesn't include code itself.
    adj_mat = np.zeros((worker_num, worker_num))
    # adj_mat = np.eye((worker_num, worker_num))
    for i in range(worker_num):
        for j in range(int(nei / 2)):
            adj_mat[i][(i - (j + 1)) % worker_num] = 1
            adj_mat[i][(i + (j + 1)) % worker_num] = 1
        # tmp = np.random.randint(worker_num - i) + i
        # adj_mat[i][tmp] = 1
        # adj_mat[tmp][i] = 1
    for i in range(worker_num):
        adj_mat[i][i] = 0
    return adj_mat


def exp2_gen_data_noise(U_gt, U_gt2, local_n, m, worker_num, snr):
    thres = 80
    beta = 10
    a = np.row_stack((thres, thres / 2, thres / 4, beta * np.random.randn(m - 3, 1)))
    b = np.array(sorted(a, reverse=True)).flatten()
    data_j = np.zeros((m, local_n))
    sigma_gt = np.diag(b)
    for i in range(local_n):
        tmp = np.random.rand(m, 1)
        tmp = tmp / norm(tmp)
        data_j[:, i] = multi_dot([U_gt + U_gt2, sigma_gt, tmp]).flatten()

    noise_data = np.zeros((data_j.shape[0], data_j.shape[1]))
    for i in range(local_n):
        noise_data[:, i] = Gnoisegen(data_j[:, i], snr).flatten()

    return data_j, noise_data, local_n
