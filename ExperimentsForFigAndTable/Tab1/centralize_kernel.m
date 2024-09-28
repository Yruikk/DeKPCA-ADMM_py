function [kernel_result] = centralize_kernel(kernel)
[m,n] = size(kernel);
kernel_result = kernel;
% if m~=n
%     return;
% end
mean_n = ones(n,n)/n;
mean_m = ones(m, m)/m;
kernel_result = kernel - kernel*mean_n - mean_m*kernel + mean_m*kernel*mean_n;
end
