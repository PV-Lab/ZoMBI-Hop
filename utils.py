import pandas as pd
import numpy as np
import sys
import itertools
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


def memory_prune(k, tolerance, X_updated, Y_updated):
    # Prune memory after each set of needle-finding zoom ins. Keep only the top k-number of points
    Y_new = np.array(Y_updated.copy()).reshape(-1) # only grab new values generated this iteration
    X_new = np.array(X_updated.copy())
#     Y_new = np.array(Y_updated[prior_n:].copy()).reshape(-1) # only grab new values generated this iteration
#     X_new = np.array(X_updated[prior_n:].copy())
    top_k = []
    top_kX = []
    while len(top_k) < k:
        ymin_index = np.argmin(Y_new)
        top_k.append(Y_new[ymin_index]) # append current best
        top_kX.append(X_new[ymin_index])
        Y_new = np.delete(Y_new, ymin_index) # delete appended best from Y_new
        X_new = np.delete(X_new, ymin_index, axis = 0) # delete from X_new as well
        similar_values = np.where((np.abs(Y_new - top_k[-1])/np.abs(top_k[-1])) < tolerance)
        Y_new = np.delete(Y_new, similar_values) # delete values similar to best from Y_new
        X_new = np.delete(X_new, similar_values, axis = 0) # delete from X_new as well
        # then repeat
        if len(Y_new) < 1:
            break # break if not enough unique values within tolerance
    df_kX = pd.DataFrame(top_kX, columns = X_updated.columns.values)[::-1] # convert to df
    df_k = pd.DataFrame(top_k, columns = Y_updated.columns.values)[::-1] # convert to df
    return df_kX, df_k


def create_penalty_mask(needle_locs, dimension_meshes, ftype, penalty_width):
    # find tolerance ranges across needle X-locations, use the 1-vector for more consistent and reliable scaling
    tol = np.abs((penalty_width * np.ones(5))).astype(ftype)
    X_upper = ((np.array(needle_locs) + tol).astype(ftype)).T
    X_lower = ((np.array(needle_locs) - tol).astype(ftype)).T
    # create binary mask, where zeros are penalty values
        # do logical AND (np.all) for all d-dimensions that fall within tolerance range => will be value of 1 if fall within tolerance
        # do logical OR (np.any) for all Gamma-number of needles => will set to 1 if a mesh value is penalized by either needle
        # take complement (~) to convert value 1 to value 0 => penalized mesh values should zero out acquisition value multiplicatively
    penalty_mask = (~np.any(np.all((dimension_meshes[:,:,None] <= X_upper[None, :, :]) & (dimension_meshes[:,:,None] >= X_lower[None, :, :]), axis=1), axis=1)).astype(int).astype(ftype).reshape(-1,1)
    return penalty_mask


def bounded_mesh(dims, lower_bound, upper_bound, ftype, resolution=10):
    # Compute GP within bounds
    dim_array = []
    for d in range(dims):
        dim_array.append(np.linspace(lower_bound[d], upper_bound[d], resolution, dtype=ftype))
    dimension_meshes = np.array(list(itertools.product(*dim_array)))
    return dimension_meshes


def m_norm(X):
    '''
    multi-dimensional normalization, takes pd dataframe as input.
    '''
    for m in range(X.shape[1]):
        X.iloc[:, m] = (X.iloc[:, m] - np.min(X.iloc[:, m])) / (np.max(X.iloc[:, m]) - np.min(X.iloc[:, m]))
    return X


def bounded_LHS(N, min_vector, max_vector):
    '''
    Runs bounded Latin Hypercube Sampling (LHS) using the iteratively zoomed-in search space bounds and memory points.
    Used to initialize the start of the forward experiments within each ZoMBI activation.
    :param N:               Number of LHS datapoints to output
    :param min_vector:      A (d,) array of lower bounds
    :param max_vector:      A (d,) array of upper bounds
    :return:                An (N,d) array of X datapoints
    '''
    samples = []
    for n in range(N):
        # randomly select points using uniform sampling within limit ranges
        samples.append(np.random.uniform(low=min_vector, high=max_vector).T)
    return np.vstack(samples)


def reset_GP(X, Y):
    # Construct GP model
    matern_hyper_tuned = ConstantKernel(1, constant_value_bounds='fixed') * Matern(length_scale=1,
                                                                                   length_scale_bounds='fixed', nu=1)
    GP = GaussianProcessRegressor(kernel=matern_hyper_tuned, n_restarts_optimizer=30, alpha=0.001, normalize_y=True)
    GP.fit(X, Y)  # Fit data to GP
    return GP


def GP_pred(X, GP_model, dtype):
    '''
    Predict f(X) means and standard deviations from data using GP.
    :param X:           Input dataset, (n,d) array
    :param GP_model:    GP regressor model
    :param dtype:       Data type to convert to, used for memory efficiency
    :return:            Predicted posterior means and standard deviations
    '''
    mean, std = GP_model.predict(X, return_std=True)
    return mean.astype(dtype), std.astype(dtype).reshape(-1, 1)  # convert to memory-efficient datatype


def initialize_arrays(X_init, Y_init):
    # initializes all storage arrays for different parts of data logging / algorithm memory
    X_intermediate = pd.DataFrame([],
                                  columns=X_init.columns.values)  # contains only intermediate points during ZoMBI activations => reset after each hop (Gamma).
    Y_intermediate = pd.DataFrame([], columns=Y_init.columns.values)

    X_all = X_init.copy().reset_index(
        drop=True)  # keep track of all data points, including all intermediate that eventually get pruned
    Y_all = Y_init.copy().reset_index(
        drop=True)  # keep track of all data points, including all intermediate that eventually get pruned

    X_GPmemory = X_all.copy()  # this is the memory that the GP has between iterations => contains initial dataset, all intermediate points until the next hop, then reset to keep only init + top-k points
    Y_GPmemory = Y_all.copy()

    X_BOUNDmemory = X_all.copy()  # this is the memory that the zooming boundary selection has between iterations => contains initial dataset, and only intermediate points of current iteration
    Y_BOUNDmemory = Y_all.copy()

    X_final = X_all.copy()  # contains only the final memory-pruned points
    Y_final = Y_all.copy()

    needles = pd.DataFrame([], columns=Y_all.columns.values)  # init df for needles to be stored
    needle_locs = pd.DataFrame([], columns=X_all.columns.values)  # init df for needle locations to be stored

    return X_intermediate, Y_intermediate, X_all, Y_all, X_GPmemory, Y_GPmemory, X_BOUNDmemory, Y_BOUNDmemory, X_final, Y_final, needles, needle_locs


def plot_penalty_dist(penalty_mask, lower_bound, upper_bound):
    plt.hist(penalty_mask)
    plt.yscale('log')
    plt.grid()
    plt.title(
        f'Percentage of Penalized Values in Current Bounds = {np.round(np.sum(~penalty_mask.astype(bool)) / len(penalty_mask) * 100, 4)}%' + '\n' + f'Upper Bound: {np.round(np.array(upper_bound), 3)}' + '\n' + f'Lower Bound: {np.round(np.array(lower_bound), 3)}')
    plt.show()
    plt.close()


def progress_bar(n, T, inc):
    '''
    Progress bar for optimization procedure.
    :param n:           The current experiment number
    :param T:           The total number of experiments to run
    :param inc:         Increment period for updating progress bar, 0.1/1 => 10 updates
    :return:            Live updating progress bar
    '''

    p = round(n / T, 2)
    total = 40
    step = int(p * total)
    block = '\u275A'
    space = '\u205F\u205F'
    if p >= inc:
        sys.stdout.write(
            f"\r{'[' + block * step + space * (total - step) + '] '}{n + 1} / {T} experiments completed . . .{' ' * step}")
        sys.stdout.flush()
        inc += 0.1
    elif (1 - inc) < 0.1 and inc <= 1:
        step = int(total)
        sys.stdout.write(f"\r{'[' + block * step + space * (total - step) + ']'} {T} / {T} Complete!{' ' * step}")
        sys.stdout.flush()
        inc += 0.1
    return inc