% Load the subject IDs from the file
subject_file = 'fMRI_data/unrelated_subjects_final.txt';

% Read subject IDs (assuming the file contains one ID per line)
subject_ids = load(subject_file);

% Loop over 280 subject and call the data_driven_fMRI function (similar to
% plot_result.m)
for i = 1:280
    subject_id = subject_ids(i);  % Get the subject ID

    % Call the data_driven_fMRI function for the current subject
    fMRI_save_matrices(subject_id);
end