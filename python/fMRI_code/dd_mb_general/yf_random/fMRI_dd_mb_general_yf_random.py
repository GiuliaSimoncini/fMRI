import numpy as np
from scipy import linalg
import pandas as pd
from control import ss

def fMRI_dd_mb_general_yf_random(subject_id):

    # Print the subject id chosen
    print(f'Subject_id: {subject_id}')

    # Get the matrices from csv
    Aest = pd.read_csv(f'../../../fMRI_data/{subject_id}/Aest.csv', header=None, delimiter=',')
    Best = pd.read_csv(f'../../../fMRI_data/{subject_id}/Best.csv', header=None, delimiter=',')
    Cest = pd.read_csv(f'../../../fMRI_data/{subject_id}/Cest.csv', header=None, delimiter=',')
    U = pd.read_csv(f'../../../fMRI_data/{subject_id}/U.csv', header=None, delimiter=',')
    Yf = pd.read_csv(f'../../../fMRI_data/{subject_id}/Yf.csv', header=None, delimiter=',')
    Y = pd.read_csv(f'../../../fMRI_data/{subject_id}/Y.csv', header=None, delimiter=',')

    # Convert to numpy arrays
    p, _ = Cest.shape  # p: output dimension (p=148 brain regions)
    m = Best.shape[1]  # m: input dimension (the inputs are divided into m=6 channels page 9)
    T = 100  # control horizon from original code
    nr = 20  # number of target states

    # Calculate Q and R
    Q = 0.001 * np.eye(14652)
    R = np.eye(600)

    # Calculate L - data-driven
    #Y: 14652 x 100, U: 600 x 100
    N = (Y.T @ Q @ Y) + (U.T @ R @ U)

    L = np.linalg.cholesky(N)

    # Calculate K_dd - data-driven
    K_dd = linalg.null_space(Yf)

    # Estimated system
    sys_est = ss(Aest, Best, Cest, np.zeros((p, m)), True)

    # Compute output controllability matrix
    Co = sys_est.C @ sys_est.B

    for jj in range(1, T):
        Co = np.hstack((Co, sys_est.C @ np.linalg.matrix_power(sys_est.A, jj) @ sys_est.B))

    # Calculate K_mb - model-based
    K_mb = linalg.null_space(Co)

    # Calculate H - output controllability matrix model-based
    H = np.zeros((p * T, m * T))

    for r in range(1, T + 1):
        for k in range(1, T + 1):
            if k > T - r:
                row_start = (r - 1) * p
                row_end = r * p
                col_start = (k - 1) * m
                col_end = k * m
                H[row_start:row_end, col_start:col_end] = sys_est.C @ np.linalg.matrix_power(sys_est.A, r - T + k - 1) @ sys_est.B

    # Calculate H_mb
    H_mb = H[:14652, :]

    # Calculate M - model-based
    P = H_mb.T @ Q @ H_mb + R

    M = np.linalg.cholesky(P)

    # Set of target states
    n_reach = nr

    # Initialize arrays for control energies and errors
    control_energy_mb = np.zeros((n_reach, 1))
    control_energy_dd = np.zeros((n_reach, 1))
    err_mb = np.zeros((n_reach, 1))
    err_dd = np.zeros((n_reach, 1))

    # Loop through each target state
    for j in range(n_reach):
        yf_random = np.random.uniform(-1, 1, (148, 1))

        # Model-based control
        u_mb = ((np.eye(600) - K_mb @ linalg.pinv(M @ K_mb) @ M) @ linalg.pinv(Co)) @ yf_random
        # Data-driven control
        u_dd = (U @ (np.eye(100) - K_dd @ (linalg.pinv(L @ K_dd)) @ L) @ linalg.pinv(Yf)) @ yf_random

        y_mb = Co @ u_mb
        y_dd = Co @ u_dd

        # Normalized error model-based
        err_mb[j] = linalg.norm(yf_random - y_mb) / p
        # Normalized error data-driven
        err_dd[j] = linalg.norm(yf_random - y_dd) / p

        # Control energy model-based
        control_energy_mb[j] = linalg.norm(u_mb)
        # Control energy data-driven
        control_energy_dd[j] = linalg.norm(u_dd)

    return control_energy_mb, control_energy_dd, err_mb, err_dd