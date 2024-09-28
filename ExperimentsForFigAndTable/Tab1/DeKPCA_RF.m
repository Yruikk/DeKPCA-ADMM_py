function [alpha,data] = DeKPCA_RF(data, nei_list, local_n, pms,adjust)
rng(0)

%% ---------------- data preparation --------------

lambda = cell(pms.worker_num,pms.worker_num);
alpha = cell(pms.worker_num,1);
alpha_ini = cell(pms.worker_num,1);
zeta  = cell(pms.worker_num,pms.worker_num);

% %% ------------compute the ground truth ------------------------
% 
% data_total =sqrt(2/pms.RFdim)*cos(omega*data_total+bias);
% data_total =  data_total - mean(data_total,2);
% tic
% kernel_tt = data_total*data_total';
% [alpha_gt,~,~,ss]= solve_global_svd(kernel_tt, 1);
% fprintf('time: %f s\n',toc);

kernel_mat = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    kernel_mat{iter} = (data{iter}- adjust{iter})*(data{iter}- adjust{iter})';
    for nei_iter = 1: length(nei_list{iter})
        nei_tmp = nei_list{iter}(nei_iter);
        lambda{iter, nei_tmp} = zeros(pms.RFdim, 1);
        zeta{iter, nei_tmp} = zeros(pms.RFdim, 1);
    end
    [alpha_ini{iter}, ~, ~,ss] = solve_global_svd(kernel_mat{iter}, 1);% a  good initial point

end
% ---------------- data preparation: end --------------

%% -------------------alpha initialization-----------------
for iter = 1: pms.worker_num
    alpha{iter} = alpha_ini{iter};% initialization
end

%% ----debug: loss variables---------
L_value = zeros(1500,1);
obj_loss = zeros(1500,1);
comm_loss = zeros(1500,1);
stage = 1;
cnt = 1;
prime_flag = 4;
%-----debug: end----------------
%
rho =  cell(pms.worker_num,1);
RHO_BASE = 10;
for iter = 1: pms.worker_num
    for nei_iter = 1: length(nei_list{iter})
        nei_tmp = nei_list{iter}(nei_iter);
        rho{iter}(nei_tmp) =  RHO_BASE;%RHO_BASE*local_n(nei_tmp)/pms.n;
    end
end

tic
for ADMM_iter = 1: 30
    %% ------------ debug: compute the lagrange function value------------
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            tmp = alpha{iter} - zeta{iter, nei_tmp};
            comm_loss(cnt) = comm_loss(cnt) + lambda{iter, nei_tmp}'*tmp;
            comm_loss(cnt) = comm_loss(cnt) + rho{iter}(nei_tmp)*(tmp'*tmp)/2;
        end
        obj_loss(cnt) = obj_loss(cnt) - norm(alpha{iter}'*data{iter}, 'fro')^2;
    end
    L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
    cnt = cnt + 1;
    %------------debug: end-----------------------
    %% ------------update zeta---------------------
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            zeta{iter, nei_tmp} = (lambda{iter, nei_tmp}/rho{iter}(nei_tmp) + lambda{nei_tmp, iter}/rho{iter}(nei_tmp) ...,
                + alpha{iter} + alpha{nei_tmp})/2;
        end
    end
    %------------ update zeta: end ---------------------

    %% ------------ debug: compute the lagrange function value------------
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            tmp = alpha{iter} - zeta{iter, nei_tmp};
            comm_loss(cnt) = comm_loss(cnt) + lambda{iter, nei_tmp}'*tmp;
            comm_loss(cnt) = comm_loss(cnt) + rho{iter}(nei_tmp)*(tmp'*tmp)/2;
        end
        obj_loss(cnt) = obj_loss(cnt) - norm(alpha{iter}'*data{iter}, 'fro')^2;
    end
    L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
    cnt = cnt + 1;
    %------------debug: end-----------------------
    if prime_flag >=4
        %% ------------ update lambda ---------------------

        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                lambda{iter, nei_tmp} = lambda{iter, nei_tmp} + ...,
                    rho{iter}(nei_tmp)*(alpha{iter} - zeta{iter, nei_tmp})/2;
            end
        end
        %----------- updata lambda: end -----------------
        %% ------------ debug: compute the lagrange function value------------
        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                tmp = alpha{iter} - zeta{iter, nei_tmp};
                comm_loss(cnt) = comm_loss(cnt) + lambda{iter, nei_tmp}'*tmp;
                comm_loss(cnt) = comm_loss(cnt) + rho{iter}(nei_tmp)*(tmp'*tmp)/2;
            end
            obj_loss(cnt) = obj_loss(cnt) - norm(alpha{iter}'*data{iter}, 'fro')^2;
        end
        L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
        cnt = cnt + 1;
        %------------debug: end-----------------------
        prime_flag = 0;
    end
    %% ------------update alpha--------------------
    for iter = 1: pms.worker_num
        tmp_sum = zeros(pms.RFdim, 1);
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            tmp_sum = tmp_sum - lambda{iter, nei_tmp} + rho{iter}(nei_tmp)*zeta{iter, nei_tmp};
        end
        KK = -2*kernel_mat{iter} + rho{iter}(iter)*length(nei_list{iter})*eye(pms.RFdim);
        %---cvx accurate solve, but slow---
        %                 cvx_begin quiet
        %                 cvx_solver sedumi
        %                     variable aa(pms.RFdim)
        %                     minimize(0.5*aa'*KK*aa -tmp_sum'*aa)
        %                     subject to
        %                         10^3*norm(aa)<=10^3
        %                 cvx_end
        % %                 if cvx_status == 'Solved' or 'Inaccurate/Solved'
        %                     alpha{iter} = aa;
        % %                 end
        %   -----(in)exact: projection gradient descent-----        
        alpha{iter} = gradient_projection_descent(KK, -tmp_sum, alpha{iter});

    end
    prime_flag = prime_flag + 1;
    %------------- update alpha: end ------------------
    %% ------------ debug: compute the lagrange function value------------
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            tmp = alpha{iter} - zeta{iter, nei_tmp};
            comm_loss(cnt) = comm_loss(cnt) + lambda{iter, nei_tmp}'*tmp;
            comm_loss(cnt) = comm_loss(cnt) + rho{iter}(nei_tmp)*(tmp'*tmp)/2;
        end
        obj_loss(cnt) = obj_loss(cnt) - norm(alpha{iter}'*data{iter}, 'fro')^2;
    end
    L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
    cnt = cnt + 1;
    %------------debug: end-----------------------
    stop_flag = 0;
    cnt_tmp = 0;
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            stop_flag =  stop_flag +  sin(subspace(alpha{iter}, zeta{iter, nei_tmp}));
            cnt_tmp = cnt_tmp + 1;
        end
    end
    if stop_flag < cnt_tmp*2e-2
%         fprintf('stop iter: %f \n. ', ADMM_iter);
        break;
    end


end

% fprintf('RF running time: %f s\n',toc);

% figure;
% plot(L_value(1:cnt-1))
end


