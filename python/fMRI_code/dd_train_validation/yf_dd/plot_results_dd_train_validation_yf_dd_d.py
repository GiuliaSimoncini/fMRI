import numpy as np
import random
import matplotlib.pyplot as plt
from python.fMRI_code.dd_train_validation.create_U_train_Yf_train import create_U_train_Yf_train
from fMRI_dd_train_validation_yf_dd import fMRI_dd_train_validation_yf_dd

# Number of subjects for train
Ntrain = 7

# Number of subjects for validation
Nvalidation = 3

# Load subject ids
unrelated_subjects = np.loadtxt('../../../fMRI_data/unrelated_subjects.txt')

# Select random subjects for training and validation without duplication
subjects_tot = random.sample(range(280), Ntrain + Nvalidation)  # Select enough subjects for both
subject_ids_train = [int(unrelated_subjects[subject]) for subject in subjects_tot[:Ntrain]]
subject_ids_validation = [int(unrelated_subjects[subject]) for subject in subjects_tot[Ntrain:]]

# Calculate U and Yf for train
U_train, Yf_train = create_U_train_Yf_train(subject_ids_train, Ntrain)

# Number of target states
nr = 20

# Initialize matrices for control energies and errors
err_dd_train_tot = np.zeros((nr, Nvalidation))
err_dd_tot = np.zeros((nr, Nvalidation))
control_energy_dd_train_tot = np.zeros((nr, Nvalidation))
control_energy_dd_tot = np.zeros((nr, Nvalidation))

# Print the subject ids used for train
print('Subject ids used for train')
for t in subject_ids_train:
    print(t)

# Print the subject ids used for validation
print('Subject ids used for validation')
for v in subject_ids_validation:
    print(v)

# Compute errors and control energies
for ii in range(Nvalidation):
    control_energy_dd, control_energy_dd_train, err_dd, err_dd_train = fMRI_dd_train_validation_yf_dd(U_train, Yf_train, subject_ids_validation[ii])
    # Flatten the arrays to make them 1D before assignment
    err_dd_train_tot[:, ii] = err_dd_train.flatten()
    err_dd_tot[:, ii] = err_dd.flatten()
    control_energy_dd_train_tot[:, ii] = control_energy_dd_train.flatten()
    control_energy_dd_tot[:, ii] = control_energy_dd.flatten()

# Plot results
plt.figure(figsize=(10, 8))

# Calculate means and standard deviations
mean_err_dd_train = np.sum(err_dd_train_tot, axis=1) / Nvalidation
std_err_dd_train = np.std(err_dd_train_tot, axis=1)
mean_err_dd = np.sum(err_dd_tot, axis=1) / Nvalidation
std_err_dd = np.std(err_dd_tot, axis=1)
mean_control_energy_dd_train = np.sum(control_energy_dd_train_tot, axis=1) / Nvalidation
std_control_energy_dd_train = np.std(control_energy_dd_train_tot, axis=1)
mean_control_energy_dd = np.sum(control_energy_dd_tot, axis=1) / Nvalidation
std_control_energy_dd = np.std(control_energy_dd_tot, axis=1)

# Title of the figure
plt.suptitle("yf data driven, subject ids all different")

# First subplot - Error
plt.subplot(2, 1, 1)
x = np.arange(nr)
width = 0.35
plt.bar(x - width/2, mean_err_dd_train, width, label='data-driven train-validation')
plt.bar(x + width/2, mean_err_dd, width, label='data-driven')
plt.xticks(x, x+1)
plt.xlabel('Target State')
plt.ylabel('Error')
plt.legend()
plt.title('Norm of the error on final output')

# Second subplot - Control Energy
plt.subplot(2, 1, 2)
plt.bar(x - width/2, mean_control_energy_dd_train, width, label='data-driven train-validation')
plt.bar(x + width/2, mean_control_energy_dd, width, label='data-driven')
plt.xticks(x, x+1)
plt.xlabel('Target State')
plt.ylabel('Energy')
plt.title('Control energy')
plt.legend()

plt.tight_layout()
plt.show()