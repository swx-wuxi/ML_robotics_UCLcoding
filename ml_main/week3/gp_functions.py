
import numpy as np
from matplotlib import pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


def fit_gp_model_1d(X_values, y_values):
    # Define the kernel
    rbf_kernel = RBF(
    length_scale=1.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e4)  # Bounds for length scale
    )
    
    # Create and fit the GP regressor
    gp = GaussianProcessRegressor(
        kernel=rbf_kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_values, y_values)
    
    return gp

def normalize_gains(X, lower, upper):
    # Map rows from the search-space bounds [lower, upper] to [0, 1]
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (np.asarray(X, dtype=float) - lower) / (upper - lower)

def make_gp_regressor(n_dims, use_white_kernel=True):
    # Amplitude * ARD RBF (+ optional white noise to absorb variation from unmodelled effects).
    # Meant to be shared by gp_minimize and the final fit; inputs are expected in [0, 1].
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 10.0)))
    if use_white_kernel:
        kernel = kernel + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )

def fit_gp_model_full(X_values, y_values, lower, upper, use_white_kernel=True):
    # X_values: (n_samples, n_dims) in original units; normalized with [lower, upper] before fitting.
    # Predict with normalize_gains(X, lower, upper).
    gp = make_gp_regressor(np.asarray(X_values).shape[1], use_white_kernel)
    gp.fit(normalize_gains(X_values, lower, upper), y_values)
    return gp

def create_prediction_range(lo, hi, n_points=200):
    return np.linspace(lo, hi, n_points).reshape(-1, 1)

def plot_gp_slices(gp_full, best_kp, best_kd, X_data, y_data, lower, upper, indices=None):
    # 1D slices of the GP: vary one parameter, hold the others at best_kp/best_kd
    best_point = np.concatenate([np.asarray(best_kp, dtype=float), np.asarray(best_kd, dtype=float)])
    n_kp = len(best_kp)
    names = [f'kp{i}' for i in range(n_kp)] + [f'kd{i}' for i in range(len(best_kd))]
    if indices is None:
        indices = range(len(best_point))
    indices = list(indices)

    ncols = min(len(indices), 4)
    nrows = int(np.ceil(len(indices) / ncols))
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for k, idx in enumerate(indices):
        lo, hi = lower[idx], upper[idx]
        lo, hi = min(lo, best_point[idx]), max(hi, best_point[idx])
        grid = np.linspace(lo, hi, 200)
        X_pred = np.tile(best_point, (len(grid), 1))
        X_pred[:, idx] = grid
        mean, std = gp_full.predict(normalize_gains(X_pred, lower, upper), return_std=True)

        plt.subplot(nrows, ncols, k + 1)
        plt.plot(grid, mean, 'k-', lw=1.5, label='Mean prediction')
        plt.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                         alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
        plt.scatter(X_data[:, idx], y_data, c='blue', s=30, edgecolors='k', label='Observations')
        plt.axvline(best_point[idx], color='r', ls='--', label='best_observed')
        plt.xlim(lo, hi)
        plt.title(f'Slice along {names[idx]}')
        plt.xlabel(names[idx])
        plt.ylabel('Tracking error')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_gp_results_1d(kp0_values, kd0_values, tracking_errors, gp_kp0, gp_kd0, kp0_range, kd0_range):
    # Create prediction ranges (kp0_range, kd0_range are (lo, hi) tuples)
    kp0_pred = create_prediction_range(*kp0_range)
    kd0_pred = create_prediction_range(*kd0_range)
    
    # Predict for kp0
    y_mean_kp0, y_std_kp0 = gp_kp0.predict(kp0_pred, return_std=True)
    
    # Predict for kd0
    y_mean_kd0, y_std_kd0 = gp_kd0.predict(kd0_pred, return_std=True)
    

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(kp0_pred, y_mean_kp0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kp0_pred.ravel(), y_mean_kp0 - 1.96 * y_std_kp0, y_mean_kp0 + 1.96 * y_std_kp0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    #plt.scatter(X, y, c='blue', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
    #plt.plot(X_pred, np.sin(X_pred), 'r:', lw=1.5, label='True function')
    plt.title("Gaussian process regression on noise-free dataset")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')
    #plt.show()




    # Plot for kp0
    #plt.figure(figsize=(14, 6))
    
    #plt.subplot(1, 2, 1)
    #plt.fill_between(
    #    kp0_pred.ravel(),
    #    y_mean_kp0 - 1.96 * y_std_kp0,
    #    y_mean_kp0 + 1.96 * y_std_kp0,
    #    alpha=0.2,
    #    label='95% Confidence Interval'
    #)
    #plt.plot(kp0_pred, y_mean_kp0, 'k-', label='GP Mean Prediction')
    #plt.scatter(kp0_values, tracking_errors, c='r', label='Data Points')
    #plt.title('GP Regression for kp0')
    #plt.xlabel('kp0')
    #plt.ylabel('Tracking Error')
    #plt.legend()
    
    # Plot for kd0
    plt.subplot(1, 2, 2)
    plt.plot(kd0_pred, y_mean_kd0, 'k-', lw=1.5, zorder=9, label='Mean prediction')
    plt.fill_between(kd0_pred.ravel(), y_mean_kd0 - 1.96 * y_std_kd0, y_mean_kd0 + 1.96 * y_std_kd0,
                    alpha=0.5, fc='orange', ec='None', label='95% confidence interval')
    #plt.scatter(X, y, c='blue', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
    #plt.plot(X_pred, np.sin(X_pred), 'r:', lw=1.5, label='True function')
    plt.title("Gaussian process regression on noise-free dataset")
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend(loc='upper left')



    #plt.fill_between(
    #    kd0_pred.ravel(),
    #    y_mean_kd0 - 1.96 * y_std_kd0,
    #    y_mean_kd0 + 1.96 * y_std_kd0,
    #    alpha=0.2,
    #    label='95% Confidence Interval'
    #)
    #plt.plot(kd0_pred, y_mean_kd0, 'k-', label='GP Mean Prediction')
    #plt.scatter(kd0_values, tracking_errors, c='r', label='Data Points')
    #plt.title('GP Regression for kd0')
    #plt.xlabel('kd0')
    #plt.ylabel('Tracking Error')
    #plt.legend()
    
    plt.tight_layout()
    plt.show()