import numpy as np
from scipy import linalg
import pandas as pd
from control import ss

def fMRI_dd_mb_yf_random(subject_id):

    # Print the subject id chosen
    print(f'Subject_id: {subject_id}')

    # Get the matrices from csv
    Aest = pd.read_csv(f'../../../fMRI_data/{subject_id}/Aest.csv', header=None, delimiter=',')
    Best = pd.read_csv(f'../../../fMRI_data/{subject_id}/Best.csv', header=None, delimiter=',')
    Cest = pd.read_csv(f'../../../fMRI_data/{subject_id}/Cest.csv', header=None, delimiter=',')
    U = pd.read_csv(f'../../../fMRI_data/{subject_id}/U.csv', header=None, delimiter=',')
    Yf = pd.read_csv(f'../../../fMRI_data/{subject_id}/Yf.csv', header=None, delimiter=',')

    # Convert to numpy arrays
    p, _ = Cest.shape  # p: output dimension (p=148 brain regions)
    m = Best.shape[1]  # m: input dimension (the inputs are divided into m=6 channels page 9)
    T = 100  # control horizon from original code
    nr = 20  # number of target states

    # Estimated system
    sys_est = ss(Aest, Best, Cest, np.zeros((p, m)), True)

    # Compute output controllability matrix
    Co = sys_est.C @ sys_est.B

    for jj in range(1, T):
        Co = np.hstack((Co, sys_est.C @ np.linalg.matrix_power(sys_est.A, jj) @ sys_est.B))

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
        u_mb = Co.T @ linalg.pinv(Co @ Co.T) @ yf_random
        # Data-driven control
        u_dd = linalg.pinv(Yf @ linalg.pinv(U)) @ yf_random

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