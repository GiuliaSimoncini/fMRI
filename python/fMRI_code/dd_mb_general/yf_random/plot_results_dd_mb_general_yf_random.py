import numpy as np
import matplotlib.pyplot as plt
import random
from fMRI_dd_mb_general_yf_random import fMRI_dd_mb_general_yf_random

# Number of subjects
N = 5

# Number of target states
nr = 20

# Load subject ids
unrelated_subjects = np.loadtxt('../../../fMRI_data/unrelated_subjects.txt')

# Choose N subjects randomly
subjects_tot = random.sample(range(280), N)

# Initialize matrices for control energies and errors
err_mb_tot = np.zeros((nr, N))
err_dd_tot = np.zeros((nr, N))
control_energy_mb_tot = np.zeros((nr, N))
control_energy_dd_tot = np.zeros((nr, N))

# Compute errors and control energies
for ii in range(N):
    subject_id = int(unrelated_subjects[subjects_tot[ii]])
    control_energy_mb, control_energy_dd, err_mb, err_dd = fMRI_dd_mb_general_yf_random(subject_id)

    # Flatten the arrays to make them 1D before assignment
    err_mb_tot[:, ii] = err_mb.flatten()
    err_dd_tot[:, ii] = err_dd.flatten()
    control_energy_mb_tot[:, ii] = control_energy_mb.flatten()
    control_energy_dd_tot[:, ii] = control_energy_dd.flatten()

# Plot results
plt.figure(figsize=(10, 8))

# Calculate means and standard deviations
mean_err_mb = np.sum(err_mb_tot, axis=1) / N
std_err_mb = np.std(err_mb_tot, axis=1)
mean_err_dd = np.sum(err_dd_tot, axis=1) / N
std_err_dd = np.std(err_dd_tot, axis=1)
mean_control_energy_mb = np.sum(control_energy_mb_tot, axis=1) / N
std_control_energy_mb = np.std(control_energy_mb_tot, axis=1)
mean_control_energy_dd = np.sum(control_energy_dd_tot, axis=1) / N
std_control_energy_dd = np.std(control_energy_dd_tot, axis=1)

# Title of the figure
plt.suptitle("yf random")

# First subplot - Error
plt.subplot(2, 1, 1)
x = np.arange(nr)
width = 0.35
plt.bar(x - width / 2, mean_err_mb, width, label='model-based')
plt.bar(x + width / 2, mean_err_dd, width, label='data-driven')
plt.xticks(x, x + 1)
plt.xlabel('Target State')
plt.ylabel('Error')
plt.legend()
plt.title('Norm of the error on final output')

# Second subplot - Control Energy
plt.subplot(2, 1, 2)
plt.bar(x - width / 2, mean_control_energy_mb, width, label='model-based')
plt.bar(x + width / 2, mean_control_energy_dd, width, label='data-driven')
plt.xticks(x, x + 1)
plt.xlabel('Target State')
plt.ylabel('Energy')
plt.title('Control energy')
plt.legend()

plt.tight_layout()
plt.show()

# N box plots for the model-based error (one for each subject)
plt.boxplot(err_mb_tot)
plt.title("err_mb_tot")
plt.show()

# N box plots for the data-driven error (one for each subject)
plt.boxplot(err_dd_tot)
plt.title("err_dd_tot")
plt.show()

# Overall box plot of the N subjects (for model-based error)
plt.boxplot(err_mb_tot.flatten())
plt.title("err_mb_tot_flatten")
plt.show()

# Overall box plot of the N subjects (for data-driven error)
plt.boxplot(err_dd_tot.flatten())
plt.title("err_dd_tot_flatten")
plt.show()

# N box plots for the model-based energy (one for each subject)
plt.boxplot(control_energy_mb_tot)
plt.title("control_energy_mb_tot")
plt.show()

# N box plots for the data-driven energy (one for each subject)
plt.boxplot(control_energy_dd_tot)
plt.title("control_energy_dd_tot")
plt.show()

# Overall box plot of the N subjects (for model-based energy)
plt.boxplot(control_energy_mb_tot.flatten())
plt.title("control_energy_mb_tot_flatten")
plt.show()

# Overall box plot of the N subjects (for data-driven energy)
plt.boxplot(control_energy_dd_tot.flatten())
plt.title("control_energy_dd_tot_flatten")
plt.show()