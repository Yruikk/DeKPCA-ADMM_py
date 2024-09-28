function [alpha_total] = method_JMLR(kernel_mat, nei_list, local_n, pms)
%% ---------------- data preparation --------------
alpha_total = cell(pms.worker_num,1);
lambda = cell(pms.worker_num,pms.worker_num, pms.worker_num);
alpha = cell(pms.worker_num,1);
alpha_ini = cell(pms.worker_num,1);
zeta  = cell(pms.worker_num,pms.worker_num);
first_eig = zeros(pms.worker_num,1);

adjust = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    adjust{iter} = zeros(local_n(iter), local_n(iter));
    alpha_total{iter} = zeros(local_n(iter), pms.target_k);
end

for k_iter = 1: pms.target_k

    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp1 =  nei_list{iter}(nei_iter);
            zeta{iter, nei_tmp1} = zeros(1, local_n(iter)) ;
            for nei_iter2 = 1: length(nei_list{iter})
                nei_tmp2 = nei_list{iter}(nei_iter2);
                lambda{nei_tmp1, nei_tmp2, iter} = zeros(1, local_n(nei_tmp1));
            end
        end
        tmp = (kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter})';
        [alpha_ini{iter}, ~, ~,first_eig(iter)] = solve_global_svd(tmp, 1);% a  good initial point
    end

    %% -------------------alpha initialization-----------------
    for iter = 1: pms.worker_num
        alpha{iter} = alpha_ini{iter};% initialization
    end
    %% ----debug: loss variables---------
    L_value = zeros(1500,1);
    obj_loss = zeros(1500,1);
    comm_loss = zeros(1500,1);
    cnt = 1;
    prime_flag = 4;
    %-----debug: end----------------
    %
    rho =  cell(pms.worker_num,1);
    RHO_BASE = 0.01/k_iter;
    for iter = 1: pms.worker_num
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            rho{iter}(nei_tmp) =  RHO_BASE*ceil(max(first_eig));
        end
    end
    %% --------invent initialization-----------
    kernel_sum = cell(pms.worker_num,1);
    for iter = 1: pms.worker_num
        tmp_kernel_sum = zeros(local_n(iter), local_n(iter));
        tmp = kernel_mat{iter,iter,iter}*kernel_mat{iter,iter,iter}';
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp = nei_list{iter}(nei_iter);
            tmp_kernel_sum = tmp_kernel_sum + rho{iter}(nei_tmp)*kernel_mat{iter, nei_tmp, iter}*kernel_mat{iter, nei_tmp, iter}' ...,
                +  rho{iter}(nei_tmp)*tmp;
        end
        %     weight = rho{iter}(iter)*length(nei_list{iter}) - 2;
        kernel_sum{iter} = tmp_kernel_sum - 2*(kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter});
    end

    tic
    for ADMM_iter = 1: 150
        %% ------------ debug: compute the lagrange function value------------
        [L_value(cnt), comm_loss(cnt), obj_loss(cnt)] = Comp_L_value(kernel_mat, nei_list, rho, pms, alpha, zeta, lambda);
        cnt = cnt + 1;
        %------------debug: end-----------------------
        %% ------------update zeta---------------------
        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                zeta{iter, nei_tmp} = alpha{iter}'*kernel_mat{iter, iter, iter}/2 ...,
                    + alpha{nei_tmp}'*kernel_mat{nei_tmp, iter, nei_tmp}/2 ...,
                    + lambda{iter, nei_tmp, iter} /rho{iter}(nei_tmp)/2 ...,
                    + lambda{iter, nei_tmp, nei_tmp} /rho{nei_tmp}(iter)/2;
            end
        end
        %------------ update zeta: end ---------------------

        %% ------------ debug: compute the lagrange function value------------
        [L_value(cnt), comm_loss(cnt), obj_loss(cnt)] = Comp_L_value(kernel_mat, nei_list, rho, pms, alpha, zeta, lambda);
        cnt = cnt + 1;
        %------------debug: end-----------------------
        
        %% ------------update alpha--------------------
        for iter = 1: pms.worker_num
            tmp_sum = zeros(local_n(iter), 1);
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                tmp_sum = tmp_sum + kernel_mat{iter, nei_tmp, iter}*(-lambda{nei_tmp, iter, iter} + rho{iter}(nei_tmp)*zeta{nei_tmp, iter})'...,
                    + kernel_mat{iter, iter, iter}*(-lambda{iter,nei_tmp, iter} + rho{iter}(nei_tmp)*zeta{iter, nei_tmp})';
            end        

            %   -----(in)exact: projection gradient descent-----
            alpha{iter} = gradient_projection_descent(kernel_sum{iter}, -tmp_sum, alpha{iter});

        end
        prime_flag = prime_flag + 1;
        %------------- update alpha: end ------------------
        %% ------------ debug: compute the lagrange function value------------
        [L_value(cnt), comm_loss(cnt), obj_loss(cnt)] = Comp_L_value(kernel_mat, nei_list, rho, pms, alpha, zeta, lambda);
        cnt = cnt + 1;
        %------------debug: end-----------------------
        if prime_flag >= 4
            %% ------------ update lambda ---------------------
            for iter = 1: pms.worker_num
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp = nei_list{iter}(nei_iter);
                    lambda{iter, nei_tmp, iter} = lambda{iter, nei_tmp, iter} + ...,
                        rho{iter}(nei_tmp)*(alpha{iter}'*kernel_mat{iter, iter, iter} - zeta{iter, nei_tmp})/2;
                    lambda{nei_tmp, iter, iter} = lambda{nei_tmp, iter, iter} + ...,
                        rho{iter}(nei_tmp)*(alpha{iter}'*kernel_mat{iter, nei_tmp, iter} - zeta{nei_tmp, iter})/2;
                end
            end
            %----------- updata lambda: end -----------------
            %% ------------ debug: compute the lagrange function value------------
            [L_value(cnt), comm_loss(cnt), obj_loss(cnt)] = Comp_L_value(kernel_mat, nei_list, rho, pms, alpha, zeta, lambda);
            cnt = cnt + 1;
            %------------debug: end-----------------------
            prime_flag = 0;
        end
        stop_flag = 0;
        cnt_tmp = 0;
        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                stop_flag =  stop_flag +  sin(subspace((alpha{iter}'* kernel_mat{iter, nei_tmp, iter})', zeta{nei_tmp, iter}'));
                stop_flag =  stop_flag +  sin(subspace((alpha{iter}'*kernel_mat{iter, iter, iter})', zeta{iter, nei_tmp}'));
                cnt_tmp = cnt_tmp + 2;
            end
        end
        if stop_flag < cnt_tmp*1e-3
%             fprintf('stop iter: %f \n. ', ADMM_iter);
            break;
        end
    end
figure;
plot(L_value(2:cnt-1))
    for iter = 1: pms.worker_num
        tmp = alpha{iter} - alpha_total{iter}*alpha_total{iter}'*kernel_mat{iter,iter,iter}*alpha{iter};
        alpha_total{iter}(:, k_iter) =  tmp/sqrt(tmp'*kernel_mat{iter,iter,iter}*tmp);
        adjust{iter} = kernel_mat{iter,iter,iter}*alpha_total{iter}*alpha_total{iter}'*kernel_mat{iter,iter,iter};
    end
end
fprintf('Method in JMLR running time: %f s\n',toc);



end

function [Lvalue, comm, obj] = Comp_L_value(kernel_mat, nei_list, rho, pms, alpha, zeta, lambda)
comm = 0;
obj = 0;
for iter = 1: pms.worker_num
    for nei_iter = 1: length(nei_list{iter})
        nei_tmp = nei_list{iter}(nei_iter);
        tmp = alpha{iter}'* kernel_mat{iter, nei_tmp, iter} - zeta{nei_tmp, iter};
        comm = comm + tmp*lambda{nei_tmp, iter, iter}';
        comm = comm + rho{iter}(nei_tmp)*(tmp*tmp')/2;
        tmp = alpha{iter}'*kernel_mat{iter, iter, iter} - zeta{iter, nei_tmp};
        comm = comm + tmp*lambda{iter, nei_tmp, iter}';
        comm = comm + rho{iter}(nei_tmp)*(tmp*tmp')/2;
    end
    obj= obj - norm(kernel_mat{iter,iter, iter}*alpha{iter}, 'fro')^2;
end
Lvalue = obj + comm;
end
