function [data, local_n, noise_data, Y] = data_generation(pms)
rng(0)
data = cell(pms.worker_num,1);
Y=[];

if pms.data_type == 0

    %% mnist
    load('mnist_all.mat')
    num_tmp  = pms.worker_num*100/2;
    X = mapminmax(double([train0(1:num_tmp,:); train9(1:num_tmp,:)]));
    Y = [zeros(num_tmp,1); ones(num_tmp,1)];

elseif pms.data_type == 1
    load('Twitter.mat');
    num_tmp  = pms.worker_num*100;
    idx = randperm(length(TwitterX), num_tmp);
    X = TwitterX(1:num_tmp,:);
    Y = TwitterY(1:num_tmp,:);
end
[pms.n, pms.m] = size(X);
% random
idx_rand = randperm(pms.n);
X = X(idx_rand,:);
Y = Y(idx_rand);

local_n = floor(pms.n/pms.worker_num)*ones(pms.worker_num,1);
%     local_n(10) = sum(local_n);
%     local_n(14) = local_n(14)/4;
local_n(end) = pms.n - sum(local_n) + local_n(end);
idx = 1;
for iter = 1: pms.worker_num
    data{iter} = double(X(idx:idx+local_n(iter) - 1, :)');
    idx = idx + local_n(iter);
end

noise_data = cell(pms.worker_num,1);
for iter = 1: pms.worker_num
    noise_data{iter} = data{iter};
    for fea_iter = 1: pms.m
        noise_data{iter}(fea_iter, :)  =  noise_data{iter}(fea_iter, :) + pms.noise_level*var(data{iter}(fea_iter, :))*rand(size(data{iter}(1,:))) ;
    end
end

