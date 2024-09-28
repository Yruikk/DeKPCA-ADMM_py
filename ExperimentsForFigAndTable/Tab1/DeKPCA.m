function [alpha_total] = DeKPCA(kernel_mat, kernel_inv, nei_list, local_n, pms)

xi = cell(pms.worker_num,1);
ee = eye(pms.worker_num);
E = cell(pms.worker_num,1);

for iter = 1: pms.worker_num
    for  nei_iter = 1:length(nei_list{iter})
        xi{iter} = [xi{iter} ee(:, nei_list{iter}(nei_iter))];
    end
    E{iter} = ones(1, length(nei_list{iter}));
end
%% ---------------- data preparation --------------
alpha_total = cell(pms.worker_num,1);
alpha_old = cell(pms.worker_num,1);
eta_old = cell(pms.worker_num,1);
eta = cell(pms.worker_num,1);
alpha = cell(pms.worker_num,1);
alpha_ini = cell(pms.worker_num,1);
z_norm = cell(1, pms.worker_num);
phi_z  = cell(pms.worker_num,pms.worker_num);
phi_z_old  = cell(pms.worker_num,pms.worker_num);
phi_z_xi = cell(pms.worker_num,1);

adjust = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    adjust{iter} = zeros(local_n(iter), local_n(iter));
    alpha_total{iter} = zeros(local_n(iter), pms.target_k);
end

for k_iter = 1: pms.target_k

    for iter = 1: pms.worker_num
        eta{iter} = zeros(local_n(iter), size(xi{iter},2));
        for nei_iter = 1: length(nei_list{iter})
            nei_tmp1 =  nei_list{iter}(nei_iter);
            phi_z_old{iter, nei_tmp1} = zeros(local_n(iter), 1);
        end
        tmp = (kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter})';
        [alpha_ini{iter}, ~, ~,first_eig(iter)] = solve_global_svd(tmp, 1);% a  good initial point
    end
    % ---------------- data preparation: end --------------

    %% -------------------alpha initialization-----------------
    for iter = 1: pms.worker_num
        alpha{iter} =alpha_ini{iter};% initialization
    end
    % -------------------alpha initialization: end -----------------

    %% ----debug: loss variables---------
    L_value = zeros(1500,1);
    obj_loss = zeros(1500,1);
    comm_loss = zeros(1500,1);
    stage = 1;
    cnt = 1;
    %-----debug: end----------------
    %
    rho =  cell(pms.worker_num,1);
    H_hat = cell(pms.worker_num,1);
    update_flag  = ones(pms.worker_num,1);
    UPDATE_THRES = 1e-3;
    STOP_FLAG = 1e-4;
    alpha_flag = 0;
    z_flag = 0;

    tic
    for ADMM_iter = 1: 150
        RHO_BASE = 1000;
        if stage == 1
            rho1 = RHO_BASE;% Random initial: 100;
            rho2 = RHO_BASE/1000;% Random initial: 10;
        elseif stage == 2
            rho1 = RHO_BASE;
            rho2 = RHO_BASE/100;
        else
            rho1 = RHO_BASE;
            rho2 = RHO_BASE;
        end
        for iter = 1: pms.worker_num
            %     rho{iter} = zeros(pms.worker_num,1);
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                if nei_tmp == iter
                    rho{iter}(nei_iter) = rho1*first_eig(iter);%local_n(nei_tmp)/pms.n;
                else
                    rho{iter}(nei_iter) = rho2*first_eig(iter);%local_n(nei_tmp)/pms.n;
                end
            end
            H_hat{iter} = 1/(sum(rho{iter}));
        end


        %% ------------update phi' z---------------------

        for iter = 1: pms.worker_num

            if update_flag(iter) > UPDATE_THRES
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp1 = nei_list{iter}(nei_iter);
                    phi_z{nei_tmp1, iter} = zeros(local_n(nei_tmp1), 1);
                    for nei_iter2 = 1: length(nei_list{iter})
                        nei_tmp2 = nei_list{iter}(nei_iter2);
                        idx_z = find(nei_list{nei_tmp2} == iter);
                        phi_z{nei_tmp1, iter}  = phi_z{nei_tmp1, iter}  + kernel_mat{nei_tmp1, nei_tmp2, iter}*...
                            (kernel_inv{nei_tmp2}*eta{nei_tmp2}(:,idx_z)+ rho{nei_tmp2}(idx_z)*alpha{nei_tmp2});
                    end
                    phi_z{nei_tmp1, iter} = H_hat{iter}*phi_z{nei_tmp1, iter};
                end
            end
        end
        %------------ update phi' z: end ---------------------

        %% ------------ update z_norm ---------------------

        z_norm_old = z_norm;
        for iter = 1: pms.worker_num
            if update_flag(iter) > UPDATE_THRES
                z_norm{iter} = ones(1, 1);
                % compute the norm
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp1 = nei_list{iter}(nei_iter);
                    idx_z = find(nei_list{nei_tmp1} == iter);
                    inf_nei1 = (kernel_inv{nei_tmp1}*eta{nei_tmp1}(:,idx_z) + alpha{nei_tmp1}*rho{nei_tmp1}(idx_z))*H_hat{iter};
                    for nei_iter2 = 1: length(nei_list{iter})
                        nei_tmp2 = nei_list{iter}(nei_iter2);
                        idx_z = find(nei_list{nei_tmp2} == iter);
                        inf_nei2 = (kernel_inv{nei_tmp2}*eta{nei_tmp2}(:,idx_z) + alpha{nei_tmp2}*rho{nei_tmp2}(idx_z))*H_hat{iter};
                        z_norm{iter} = z_norm{iter} + diag(inf_nei1'*kernel_mat{nei_tmp1, nei_tmp2, iter}*inf_nei2);
                    end
                end
                z_norm{iter} = sqrt(max(z_norm{iter}, ones(1,1)));

                % divide the norm.
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp = nei_list{iter}(nei_iter);
                    phi_z{nei_tmp, iter} = phi_z{nei_tmp, iter}*diag(1./z_norm{iter});
                end
            end
        end

        % phi_j z_p should have the same direction as phi_j z_j.
        for iter = 1: pms.worker_num
            if update_flag(iter) > UPDATE_THRES
                for nei_iter = 1: length(nei_list{iter})
                    nei_tmp = nei_list{iter}(nei_iter);
                    direction = diag(phi_z{iter, nei_tmp}'*phi_z{iter, iter});
                    phi_z{iter, nei_tmp} = phi_z{iter, nei_tmp}*diag(sign(direction));
                end
            end
        end
        %------------ update z_norm : end ---------------------

        %% --------------- compute the convergence of z ---------
        delta_z = 0;
        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                delta_z = delta_z + norm(phi_z{iter, nei_tmp} - phi_z_old{iter, nei_tmp});
            end
        end
        if  delta_z < pms.worker_num*UPDATE_THRES % if z converges, then update eta.
            z_flag = 1;
        end
        phi_z_old =  phi_z;

        % --------------- convergence of z: end ----------------

        %% ------------ compute phi z xi --------------------
        for iter =  1: pms.worker_num
            phi_z_xi{iter}  = [phi_z{iter,iter}];
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp = nei_list{iter}(nei_iter);
                if nei_tmp  ~= iter
                    phi_z_xi{iter} = [phi_z_xi{iter} phi_z{iter, nei_tmp}];
                end
            end
        end
        % ------------------ phi_z_xi: end --------------------

%         %% ------------ debug: compute the lagrange function value------------
% 
%         for iter = 1: pms.worker_num
%             alpha_minus_z =(alpha{iter}*E{iter} - kernel_inv{iter}* phi_z_xi{iter} );
%             obj_loss(cnt) = obj_loss(cnt) - alpha{iter}'*(kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter})*alpha{iter};
%             comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
%             comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*kernel_mat{iter,iter,iter}*alpha_minus_z*diag(rho{iter}));
%         end
%         L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
%         cnt = cnt + 1;
        %------------debug: end-----------------------

        %% ------------update alpha---------------------
        delta_alpha = 0;
        for iter = 1: pms.worker_num
            alpha_old{iter} = alpha{iter};
            H_alpha = -2* (kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter}) ...,
                + sum(rho{iter})*1*eye(local_n(iter));
            f_alpha = (kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter}) - eta{iter})*E{iter}';

            tmp_a = alpha{iter};
            for ii =1: 10
                grad = H_alpha*tmp_a -f_alpha; % calculate gradient
                lr = diag(grad'*grad) ./ diag(grad'*H_alpha*grad);% calculate step size alpha
                tmp_a_new = tmp_a - lr * grad;% take gradient step
                if norm(tmp_a_new - tmp_a) < 1e-3 ...,
                        ||  0.5*tmp_a'*H_alpha*tmp_a -f_alpha'*tmp_a < ...,
                        0.5*tmp_a_new'*H_alpha*tmp_a_new -f_alpha'*tmp_a_new% check for convergence
                    break
                end
                tmp_a = tmp_a_new;% update x
            end
            alpha{iter} = tmp_a;
            delta_alpha = delta_alpha + norm(alpha{iter} - alpha_old{iter});
        end
        if  delta_alpha < pms.worker_num*UPDATE_THRES % if alpha converges, then update eta.
            alpha_flag = 1;
        end
        %------------- update alpha: end ------------------
                %% ------------debug: compute the lagrange function value------------
        
                for iter = 1: pms.worker_num
                    alpha_minus_z =(alpha{iter}*E{iter}- kernel_inv{iter}*phi_z_xi{iter});
                    obj_loss(cnt) = obj_loss(cnt) - alpha{iter}'*(kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter})*alpha{iter};
                    comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
                    comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*kernel_mat{iter,iter,iter}*alpha_minus_z*diag(rho{iter}));
                end
                L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
                cnt = cnt + 1;
        %------------debug: end-----------------------


        %% ------------ update eta ---------------------
        if z_flag && alpha_flag
            eta_tmp=0;
            for iter = 1: pms.worker_num
                eta_old{iter} = eta{iter};
                eta{iter} = eta{iter}  + (alpha{iter}*E{iter} - kernel_inv{iter}*phi_z_xi{iter})*diag(rho{iter});
                update_flag(iter) = sin(subspace(alpha{iter}*E{iter}, kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter})));
                eta_tmp =  eta_tmp + norm(eta{iter});
            end
            if sum(update_flag) < pms.worker_num*STOP_FLAG
                stage = stage + 1;
%                                             fprintf('iter: %d, stage:%d\n',ADMM_iter, stage);
                if stage == 4
                    break;
                end
            end
            alpha_flag = 0;
            z_flag = 0;
            %----------- updata eta: end -----------------
        end
        %% ------------ compute the update flag of z -------------
        for iter = 1: pms.worker_num
            update_flag(iter) = sin(subspace(alpha{iter}*E{iter}, kernel_inv{iter}*phi_z_xi{iter}*diag(rho{iter})));
        end
        % ------------------- update flag of z: end --------------------
%                 %% ------------debug: compute the lagrange function value------------
%         
%                 for iter = 1: pms.worker_num
%                     alpha_minus_z =(alpha{iter}*E{iter}- kernel_inv{iter}*phi_z_xi{iter});
%                     obj_loss(cnt) = obj_loss(cnt) - alpha{iter}'*(kernel_mat{iter, iter, iter} - adjust{iter})*(kernel_mat{iter,iter,iter} - adjust{iter})*alpha{iter};
%                     comm_loss(cnt) = comm_loss(cnt) + trace(eta{iter}'*alpha_minus_z);
%                     comm_loss(cnt) = comm_loss(cnt) + 0.5*trace(alpha_minus_z'*kernel_mat{iter,iter,iter}*alpha_minus_z*diag(rho{iter}));
%                 end
%                 L_value(cnt) = obj_loss(cnt) + comm_loss(cnt);
%                 cnt = cnt + 1;
%         %------------debug: end-----------------------
    end
    % %
%             figure;
%             plot(L_value(2:cnt-1),'ro-')

    for iter = 1: pms.worker_num
        tmp = alpha{iter} - alpha_total{iter}*alpha_total{iter}'*kernel_mat{iter,iter,iter}*alpha{iter};
        alpha_total{iter}(:, k_iter) =  tmp/sqrt(tmp'*kernel_mat{iter,iter,iter}*tmp);
        adjust{iter} = kernel_mat{iter,iter,iter}*alpha_total{iter}*alpha_total{iter}'*kernel_mat{iter,iter,iter};
    end
end


fprintf('DeKPCA running time: %f s\n',toc);

%


end