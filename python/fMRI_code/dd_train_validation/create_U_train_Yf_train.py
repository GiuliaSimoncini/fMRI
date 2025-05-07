import numpy as np
import pandas as pd

def create_U_train_Yf_train(list_of_subjects_for_train, Ntrain):
    U_train = []  # List to store the U arrays
    Yf_train = []  # List to store the Yf arrays

    for i in range(Ntrain):
        # Read the CSV files for each subject
        U_temp = pd.read_csv(f'../../../fMRI_data/{list_of_subjects_for_train[i]}/U.csv', header=None, delimiter=',')
        Yf_temp = pd.read_csv(f'../../../fMRI_data/{list_of_subjects_for_train[i]}/Yf.csv', header=None, delimiter=',')

        # Append each subject's U and Yf data to the lists
        U_train.append(U_temp.to_numpy())  # Convert to numpy array before appending
        Yf_train.append(Yf_temp.to_numpy())

    # Stack the lists into 2D arrays
    U_train = np.hstack(U_train)  # Stack all U arrays horizontally
    Yf_train = np.hstack(Yf_train)  # Stack all Yf arrays horizontally

    return U_train, Yf_train