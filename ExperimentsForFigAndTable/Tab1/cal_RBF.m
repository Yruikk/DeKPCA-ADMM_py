function [kernel] = cal_RBF(X, Y, sigma)
kernel = zeros(size(X,2),size(Y,2));
AA = sum(X.^2);
BB = sum(Y.^2);
tmp1 = AA' *ones(1, size(BB, 2));
tmp2 = ones(size(AA, 2), 1)*BB;
kernel = tmp1 + tmp2 - 2*X'*Y;
kernel = exp(-kernel./(sigma^2));
% kernel=kernel./length(kernel);
end

function [val] = cal_RBF_fun(x,y, sigma)
val = 1*exp(-norm(x-y,2)^2/sigma^2);
end