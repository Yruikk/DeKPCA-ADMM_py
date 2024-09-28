function [data, label, local_n, X_test, Y_test] = data_generation_B(pms)
% rng(10)
data = cell(pms.worker_num,1);
label = cell(pms.worker_num,1);

if pms.data_type == 0

    %% mnist2
    load('mnist_all.mat')
    num_train  = pms.worker_num*pms.train_num/2;
    num_test = pms.test_num/2;
    idx = randperm(min(length(train1), length(train9)),num_train+num_test);
    idx_train = idx(1:num_train);
    idx_test = setdiff(idx, idx_train);

    X_train = mapminmax( double([train1(idx_train,:); train9(idx_train,:)])');
    Y_train = [-1*ones(num_train,1); ones(num_train,1)]';
    X_test = mapminmax(double( [train1(idx_test,:); train9(idx_test,:)])');
    Y_test = [-1*ones(num_test,1); ones(num_test,1)]';


elseif pms.data_type == 1
    load('Twitter.mat');
    num_train  = pms.worker_num*pms.train_num;
    num_test = pms.test_num;
    idx = randperm(length(TwitterX), num_train+num_test);
    idx_train = idx(1:num_train);
    idx_test = setdiff(idx, idx_train);
    TwitterX = mapminmax(double(TwitterX'));
    X_train = TwitterX(:,idx_train);
    X_test =  TwitterX(:,idx_test);

    YY = mapminmax(log(double(TwitterY'+1))+1);
    %     YY = mapminmax(double(YY'));
    Y_train = YY(idx_train);
    Y_test = YY(idx_test);

elseif pms.data_type == 2
    load('TomsHardware.mat')
    XX = X;     YY = Y;

    num_train  = pms.worker_num*pms.train_num;
    num_test = pms.test_num;
    idx = randperm(length(XX),num_train+num_test);
    idx_train = idx(1:num_train);
    idx_test = setdiff(idx, idx_train);
    XX = mapminmax(double(XX'));
    X_train = XX(:,idx_train);
    X_test =  XX(:,idx_test);
    YY = mapminmax(log(double(Y'+1))+1);
    %     YY = mapminmax(double(YY'));
    Y_train = YY(idx_train);
    Y_test = YY(idx_test);

    %     X_train = mapminmax( double(XX(idx_train,:))');
    %     X_test = mapminmax(double( XX(idx_test,:))');
    %     norm_train = max(abs(YY(idx_train)'));
    %     Y_train = YY(idx_train)'/norm_train;
    %     Y_test = YY(idx_test)'/norm_train;

end

[pms.m, pms.n] = size(X_train);

if pms.divide_mode ==0
    % random
    idx_rand = randperm(pms.n);
    X_train = X_train(:,idx_rand);
    Y_train = Y_train(idx_rand);

    local_n = floor(pms.n/pms.worker_num)*ones(pms.worker_num,1);
    local_n(end) = pms.n - sum(local_n) + local_n(end);
    idx = 1;
    for iter = 1: pms.worker_num
        data{iter} = [X_train(:,idx:idx+local_n(iter) - 1)  X_test];
        label{iter} = [double(Y_train(idx:idx+local_n(iter) - 1)') ; double(Y_test')];
        idx = idx + local_n(iter);
    end
elseif pms.divide_mode == 1
    idx_rand = randperm(pms.n);
    X_train = X_train(:,idx_rand);
    Y_train = Y_train(idx_rand);

    local_n = floor(pms.n/pms.worker_num)*ones(pms.worker_num,1);
    tmp_a = 50;
    local_n = floor((pms.n-3*tmp_a)/(pms.worker_num-3))*ones(pms.worker_num,1);
    local_n(5) = tmp_a;local_n(4) = tmp_a;local_n(6) = tmp_a;
    local_n(end) = pms.n - sum(local_n) + local_n(end);
    idx = 1;
    for iter = 1: pms.worker_num
        data{iter} = [X_train(:,idx:idx+local_n(iter) - 1)  X_test];
        label{iter} = [double(Y_train(idx:idx+local_n(iter) - 1)') ; double(Y_test')];
        idx = idx + local_n(iter);
    end
elseif pms.divide_mode == 2
    idx_rand = randperm(pms.n);
    X_train = X_train(:,idx_rand);
    Y_train = Y_train(idx_rand);

    local_n = 500*ones(pms.worker_num,1) + randi(1000, pms.worker_num,1);
    local_n(end) = pms.n - sum(local_n) + local_n(end);
    idx = 1;
    for iter = 1: pms.worker_num
        data{iter} = [X_train(:,idx:idx+local_n(iter) - 1)  X_test];
        label{iter} = [double(Y_train(idx:idx+local_n(iter) - 1)') ; double(Y_test')];
        idx = idx + local_n(iter);
    end
end

local_n = local_n  + size(X_test, 2);
Y_test = double(Y_test');

