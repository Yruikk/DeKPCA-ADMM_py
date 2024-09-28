import numpy as np


# Define the objective function
def objective(x, A, b):
    return 0.5 * (x.T @ A @ x) + (b.T @ x)


# Define the gradient of the objective function
def gradient(x, A, b):
    return (A @ x) + b


# define the projection operator
def projection(x):
    norm_x = np.linalg.norm(x)
    if norm_x <= 1:
        return x
    else:
        return x / norm_x


def gradient_projection_descent(A, b, x_init, max_iter=1000, tol=1e-6):
    # Initialize the variables
    x = x_init

    for i in range(max_iter):
        # calculate gradient
        grad = gradient(x, A, b)

        # calculate step size alpha
        alpha = np.linalg.norm(grad) ** 2 / (grad.T @ A @ grad)

        # take gradient step
        x_new = x - alpha * grad

        # project onto feasible set
        x_new = projection(x_new)

        # check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        # update x
        x = x_new

    return x_new


if __name__ == '__main__':
    # Define the problem data
    # Generate a random matrix
    n = 100  # size of the matrix
    AA = np.eye(n)
    BB = np.random.rand(n)
    x = gradient_projection_descent(AA, BB)
    print(x)
    print(BB)
