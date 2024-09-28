function [x_new] = gradient_projection_descent(KK, BB, x_ini)

% Initialize the variables
x = x_ini;
max_iter = 100;
tol = 1e-4;
x_new = x;
for iter =1: max_iter

    % calculate gradient
    grad = gradient(x, KK, BB);

    % calculate step size alpha
    alpha = min(1e-3,  norm(grad)^2 / (grad'*KK*grad));    
    if norm(alpha) <1e-5
        break;
    end

    % take gradient step
    x_new = x - alpha * grad;

    % project onto feasible set
    x_new = projection(x_new);

    % check for convergence
    if norm(x_new - x) < tol     
        break
    end
    % update x
    x = x_new;
end

end

%% define the objective function
function [tt] = objective(x, A, b)
tt = 0.5 * x'* A*x + b'*x;
end

%% define  the gradient of the objective function
function [tt] = gradient(x, A, b)
tt = A*x + b;
end

%% define  the projection operator
function [tt] = projection(x)
if norm(x) <= 1
    tt = x;
else
    tt =  x / norm(x);
end
end