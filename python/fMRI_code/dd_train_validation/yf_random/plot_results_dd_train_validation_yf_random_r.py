import numpy as np
import random
import matplotlib.pyplot as plt
from python.fMRI_code.dd_train_validation.create_U_train_Yf_train import create_U_train_Yf_train
from fMRI_dd_train_validation_yf_random import fMRI_dd_train_validation_yf_random

# Number of subjects for train
Ntrain = 6

# Number of subjects for validation
Nvalidation = 3

# Load subject ids
unrelated_subjects = np.loadtxt('../../../fMRI_data/unrelated_subjects.txt')

# Subjects used for train
subject_ids_train = [0] * Ntrain #The various subjects with which we construct U and Yf
subjects_tot_train = random.sample(range(280), Ntrain)
for i in range(Ntrain):
    subject_ids_train[i] = int(unrelated_subjects[subjects_tot_train[i]])

# Subjects used for validation
subject_ids_validation = [0] * Nvalidation
subjects_tot_validation = random.sample(range(280), Nvalidation)
for i in range(Nvalidation):
    subject_ids_validation[i] = int(unrelated_subjects[subjects_tot_validation[i]])

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
print('Subject ids for train')
for t in subject_ids_train:
    print(t)

# Print the subject ids used for validation
print('Subject ids for validation')
for v in subject_ids_validation:
    print(v)

# Compute errors and control energies
for ii in range(Nvalidation):
    control_energy_dd, control_energy_dd_train, err_dd, err_dd_train = fMRI_dd_train_validation_yf_random(U_train, Yf_train, subject_ids_validation[ii])
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
plt.suptitle("yf random, subject ids random")

# First subplot - Error
plt.subplot(2, 1, 1)
x = np.arange(nr)
width = 0.35
plt.bar(x - width/2, mean_err_dd_train, width, label='data_driven train-validation')
plt.bar(x + width/2, mean_err_dd, width, label='data-driven')
plt.xticks(x, x+1)
plt.xlabel('Target State')
plt.ylabel('Error')
plt.legend()
plt.title('Norm of the error on final output')

# Second subplot - Control Energy
plt.subplot(2, 1, 2)
plt.bar(x - width/2, mean_control_energy_dd_train, width, label='data_driven train-validation')
plt.bar(x + width/2, mean_control_energy_dd, width, label='data-driven')
plt.xticks(x, x+1)
plt.xlabel('Target State')
plt.ylabel('Energy')
plt.title('Control energy')
plt.legend()

plt.tight_layout()
plt.show()

# N box plots for the data-driven error (one for each subject)
plt.boxplot(err_dd_tot)
plt.title("err_dd_tot")
plt.show()

# N box plots for the data-driven train-validation error (one for each subject)
plt.boxplot(err_dd_train_tot)
plt.title("err_dd_train_tot")
plt.show()

# Overall box plot of the N subjects (for data-driven error)
plt.boxplot(err_dd_tot.flatten())
plt.title("err_dd_tot_flatten")
plt.show()

# Overall box plot of the N subjects (for data-driven train-validation error)
plt.boxplot(err_dd_train_tot.flatten())
plt.title("err_dd_train_tot_flatten")
plt.show()

# N box plots for the data-driven energy (one for each subject)
plt.boxplot(control_energy_dd_tot)
plt.title("control_energy_dd_tot")
plt.show()

# N box plots for the data-driven train-validation energy (one for each subject)
plt.boxplot(control_energy_dd_train_tot)
plt.title("control_energy_dd_train_tot")
plt.show()

# Overall box plot of the N subjects (for data-driven energy)
plt.boxplot(control_energy_dd_tot.flatten())
plt.title("control_energy_dd_tot_flatten")
plt.show()

# Overall box plot of the N subjects (for data-driven train-validation energy)
plt.boxplot(control_energy_dd_train_tot.flatten())
plt.title("control_energy_dd_train_tot_flatten")
plt.show()

# Box plot comparing err_dd_tot.flatten() and err_dd_train_tot.flatten() in the same figure
plt.figure(figsize=(6, 5))
plt.boxplot([err_dd_tot.flatten(), err_dd_train_tot.flatten()], tick_labels=["DD", "DD_train_validation"])
plt.title("err_dd_tot vs err_dd_train_tot")
plt.ylabel("Error")
plt.show()

# Box plot comparing control_energy_dd_tot_flatten and control_energy_dd_train_tot_flatten in the same figure
plt.figure(figsize=(6, 5))
plt.boxplot([control_energy_dd_tot.flatten(), control_energy_dd_train_tot.flatten()], tick_labels=["DD", "DD_train_validation"])
plt.title("control_energy_dd_tot_flatten vs control_energy_dd_train_tot_flatten")
plt.ylabel("Control energy")
plt.show()