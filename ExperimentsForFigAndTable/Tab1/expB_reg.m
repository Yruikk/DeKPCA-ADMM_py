clear all; close all;

variable_j = [10];
pms.target_k = 15;
pms.train_num = 500;
pms.test_num = 100;
pms.data_type = 2;
pms.divide_mode =0;
max_repeat = 1;
test_result= [];
for j_iter = 1: length(variable_j)
    pms.worker_num = variable_j(j_iter);
    % ----------------compute the digragh from the undigraph--------------
    adj_mat = eye(pms.worker_num);
    nei_list = cell(pms.worker_num,1);
    for iter = 1: pms.worker_num-1
        adj_mat(iter, iter +1) = 1;
        adj_mat(iter+1, iter) = 1;
        %         tmp = randi((pms.worker_num - iter), 1,1) + iter;
        %         adj_mat(iter, tmp) = ones(1, 1);
        %         adj_mat(tmp, iter) = ones(1, 1);
    end
    adj_mat(pms.worker_num, 1) = 1;
    adj_mat(1, pms.worker_num) = 1;
    for iter = 1: pms.worker_num
        nei_list{iter}= unique([iter find(adj_mat(iter, :)==1)], 'stable');
    end

    acc_ini = zeros(pms.worker_num, max_repeat);
    acc_DeKPCA = zeros(pms.worker_num, max_repeat);
    acc_JMLR = zeros(pms.worker_num, max_repeat);
    acc_RF = zeros(pms.worker_num, max_repeat);

    for repeat = 1: max_repeat
        rng(repeat)
        [data, label, local_n, X_test, Y_test] = data_generation_B(pms);%the last pms.test_num samples are used for test.
        pms.n = sum(local_n);
        pms.m = size(X_test, 1);
        pms.test_num = size(X_test, 2);
        pms.sigma = sqrt(pms.m)/0.5; %hyper-parameter of RBF
        Y_MAX = max(label{1})+10;
        Y_MIN = min(label{1})-10;

        %% ------------- kernel matrix preapre ------------

        kernel_mat = cell(pms.worker_num, pms.worker_num, pms.worker_num);
        kernel_inv = cell(pms.worker_num, 1);
        for iter = 1: pms.worker_num
            for nei_iter = 1: length(nei_list{iter})
                nei_tmp1 =  nei_list{iter}(nei_iter);
                for nei_iter2 = nei_iter: length(nei_list{iter})
                    nei_tmp2 = nei_list{iter}(nei_iter2);
                    kernel_mat{nei_tmp1,nei_tmp2, iter} = cal_RBF(data{nei_tmp1}, data{nei_tmp2}, pms.sigma);
                    [kernel_mat{nei_tmp1,nei_tmp2, iter}] = centralize_kernel(kernel_mat{nei_tmp1,nei_tmp2, iter}); % centralization
                    kernel_mat{nei_tmp2, nei_tmp1, iter} = kernel_mat{nei_tmp1,nei_tmp2, iter}';
                end
            end

            [alpha_ini{iter}, ~, ~,lam_ini{iter}] = solve_global_svd(kernel_mat{iter, iter, iter}, pms.target_k);% a  good initial point
            ill_thres = 0.01;
            kernel_mat{iter,iter, iter} = kernel_mat{iter,iter,iter} + ill_thres*min(lam_ini{iter})/local_n(iter)*ones(size(kernel_mat{iter,iter,iter}));
            % !!! directly take inv on K will lead to undesirable numerical problem
            [v, d] = eig(0.5*(kernel_mat{iter,iter, iter} + kernel_mat{iter,iter, iter}'));
            dd= diag(d);
            [idx_pos] = find(dd > 1e-3);
            dd(idx_pos) = 1./(dd(idx_pos));
            kernel_inv{iter} = v*diag(dd)*v';

        end

        %% --------------ini-----------------------
        for iter = 1: pms.worker_num
            new_data = (kernel_mat{iter,iter,iter}*alpha_ini{iter})'; %data after feature reduction
            idx_tmp = local_n(iter)-pms.test_num;
            new_train = new_data(:,1:idx_tmp);
            new_test = new_data(:, idx_tmp+1: end);
            net = feedforwardnet([2*pms.target_k 10]); 
            [net,tr] = train(net,new_train, label{iter}(1:idx_tmp)');
            Y_pre = max(Y_MIN, min(Y_MAX, net(new_test)))';
            acc_ini(iter, repeat) = norm(Y_pre - Y_test)^2/norm(Y_test - mean(Y_test))^2;
        end

        %% ------------ DeKPCA -----------------
        [alpha_DeKPCA] = DeKPCA(kernel_mat, kernel_inv, nei_list, local_n, pms);

        for iter = 1: pms.worker_num
            new_data = (kernel_mat{iter,iter,iter}*alpha_DeKPCA{iter})'; %data after feature reduction
            idx_tmp = local_n(iter)-pms.test_num;
            new_train = new_data(:,1:idx_tmp);
            new_test = new_data(:, idx_tmp+1: end);
            net = feedforwardnet([2*pms.target_k 10]); 
            [net,tr] = train(net,new_train, label{iter}(1:idx_tmp)');
            Y_pre = max(Y_MIN, min(Y_MAX, net(new_test)))';
            acc_DeKPCA(iter, repeat) = norm(Y_pre - Y_test)^2/norm(Y_test - mean(Y_test))^2;
        end

        %% ------------ JMLR 2010 ----------------

        [alpha_JMLR] = method_JMLR(kernel_mat, nei_list, local_n, pms);

        for iter = 1: pms.worker_num
            new_data = (kernel_mat{iter,iter,iter}*alpha_JMLR{iter})'; %data after feature reduction
            idx_tmp = local_n(iter)-pms.test_num;
            new_train = new_data(:,1:idx_tmp);
            new_test = new_data(:, idx_tmp+1: end);
            net_wid = max(10, 2*pms.target_k);
            net = feedforwardnet(net_wid);
            [net,tr] = train(net,new_train, label{iter}(1:idx_tmp)');
            Y_pre = max(Y_MIN, min(Y_MAX, net(new_test)))';
            acc_JMLR(iter, repeat) = norm(Y_pre - Y_test)^2/norm(Y_test - mean(Y_test))^2;
        end

        %% ------------- RF--------------------
        %% -------------compute the random features --------------------
        data_dim = size(data{1}(:,1), 1);
        data_RF = cell(pms.worker_num,1);
        pms.RFdim = ceil(data_dim *3.5); % number of random features.
        omega = (1/pms.sigma)*randn(pms.RFdim, data_dim);
        bias = rand(pms.RFdim, 1)*2*pi;
        pms.n = sum(local_n);
        data_total =  [];
        for iter  = 1:pms.worker_num
            %     data_total = [data_total data{iter}];
            data_RF{iter} = sqrt(2/pms.RFdim)*cos(omega*data{iter}+bias);
            data_RF{iter} = data_RF{iter} - mean(data_RF{iter},2);
        end
        %% --- finish -------
        adjust = cell(pms.worker_num,1);
        alpha_RF = cell(pms.worker_num,1);
        for iter = 1: pms.worker_num
            alpha_RF{iter}=zeros(pms.RFdim, pms.target_k);
            adjust{iter} = zeros(size(data_RF{iter}));
            lam_RF{iter} = zeros(pms.target_k,1);
        end
        for k_iter = 1: pms.target_k
            [tmp_RF] = DeKPCA_RF(data_RF, nei_list, local_n, pms, adjust);
            for iter = 1: pms.worker_num
                tmp_RF{iter} = tmp_RF{iter} - alpha_RF{iter}*alpha_RF{iter}'*tmp_RF{iter};
                tmp_RF{iter} = tmp_RF{iter}/sqrt(tmp_RF{iter}'*tmp_RF{iter});
                alpha_RF{iter}(:, k_iter) = tmp_RF{iter};
                lam_RF{iter}(k_iter) = tmp_RF{iter}'*data_RF{iter}*data_RF{iter}'*tmp_RF{iter};
                adjust{iter} = alpha_RF{iter}*alpha_RF{iter}'*data_RF{iter};
            end
        end
        %         [alpha_RF, data_RF] = DeKPCA_RF(data, nei_list, local_n, pms);

        for iter = 1: pms.worker_num
            new_data = alpha_RF{iter}'*data_RF{iter}; %data after feature reduction
            idx_tmp = local_n(iter)-pms.test_num;
            new_train = new_data(:,1:idx_tmp);
            new_test = new_data(:, idx_tmp+1: end);
            net = feedforwardnet([2*pms.target_k 10]); 
            [net,tr] = train(net,new_train, label{iter}(1:idx_tmp)');
            Y_pre = max(Y_MIN, min(Y_MAX, net(new_test)))';
            acc_RF(iter, repeat) = norm(Y_pre - Y_test)^2/norm(Y_test - mean(Y_test))^2;
        end
    end
    mean_acc_ini = mean(abs(acc_ini),2);
    mean_acc_DeKPCA = mean(abs(acc_DeKPCA),2);
    mean_acc_JMLR = mean(abs(acc_JMLR),2);
    mean_acc_RF = mean(abs(acc_RF),2);
    fprintf('Local mean: %f, std: %f\n', mean(mean_acc_ini),std(mean_acc_ini));
    fprintf('DeKPCA mean: %f, std: %f\n', mean(mean_acc_DeKPCA),std(mean_acc_DeKPCA));
    fprintf('JMLR mean: %f, std: %f\n', mean(mean_acc_JMLR),std(mean_acc_JMLR));
    fprintf('RF mean: %f, std: %f\n', mean(mean_acc_RF),std(mean_acc_RF));
    figure; hold on
    plot(mean_acc_ini, 'go', MarkerSize=8, DisplayName='Ini')
    plot(ones(1,length(mean_acc_ini))*mean(mean_acc_ini),'g-',LineWidth=3, DisplayName='Ini')
    plot(mean_acc_DeKPCA, 'r*', MarkerSize=8, DisplayName='Ours')
    plot(ones(1,length(mean_acc_DeKPCA))*mean(mean_acc_DeKPCA),'r:',LineWidth=3, DisplayName='Ours')
    %     plot(mean_acc_JMLR, 'b^', MarkerSize=8, DisplayName='JMLR')
    %     plot(ones(1,length(mean_acc_JMLR))*mean(mean_acc_JMLR),'b:',LineWidth=3, DisplayName='JMLR')
    plot(mean_acc_RF, 'k.', MarkerSize=8, DisplayName='RF')
    plot(ones(1,length(mean_acc_RF))*mean(mean_acc_RF),'k:',LineWidth=3, DisplayName='RF')
    legend('show', 'Location','best')
end

