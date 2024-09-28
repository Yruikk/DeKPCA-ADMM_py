function [W, tr, u, ss]  = solve_global_svd(cor_mat_noise, target_k)

[u, s, v] = svd(cor_mat_noise);
tr = trace(s(1:target_k, 1:target_k));
W = v(:,1:target_k);
s = diag(s);
ss = s(1: target_k);
end