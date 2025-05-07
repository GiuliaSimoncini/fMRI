import numpy as np
from scipy import linalg
import pandas as pd
from control import ss

def fMRI_dd_train_validation_yf_random(U_train, Yf_train, subject_id_validation):

    # Get the matrices from csv
    Aest = pd.read_csv(f'../../../fMRI_data/{subject_id_validation}/Aest.csv', header=None, delimiter=',')
    Best = pd.read_csv(f'../../../fMRI_data/{subject_id_validation}/Best.csv', header=None, delimiter=',')
    Cest = pd.read_csv(f'../../../fMRI_data/{subject_id_validation}/Cest.csv', header=None, delimiter=',')
    U = pd.read_csv(f'../../../fMRI_data/{subject_id_validation}/U.csv', header=None, delimiter=',')
    Yf = pd.read_csv(f'../../../fMRI_data/{subject_id_validation}/Yf.csv', header=None, delimiter=',')

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
    control_energy_dd = np.zeros((n_reach, 1))
    control_energy_dd_train = np.zeros((n_reach, 1))
    err_dd = np.zeros((n_reach, 1))
    err_dd_train = np.zeros((n_reach, 1))

    # Loop through each target state
    for j in range(n_reach):
        yf_random = np.random.uniform(-1, 1, (148, 1))

        # Data-driven control
        u_dd = linalg.pinv(Yf @ linalg.pinv(U)) @ yf_random
        # Data-driven train-validation control
        u_dd_train = linalg.pinv(Yf_train @ linalg.pinv(U_train)) @ yf_random

        y_dd = Co @ u_dd
        y_dd_train = Co @ u_dd_train

        # Normalized error data-driven
        err_dd[j] = linalg.norm(yf_random - y_dd) / p
        # Normalized error data-driven train-validation
        err_dd_train[j] = linalg.norm(yf_random - y_dd_train) / p

        # Control energy data-driven
        control_energy_dd[j] = linalg.norm(u_dd)
        # Control energy data-driven train-validation
        control_energy_dd_train[j] = linalg.norm(u_dd_train)

    return control_energy_dd, control_energy_dd_train, err_dd, err_dd_train