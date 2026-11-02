%% DLRS_CVR_BAT - Original SPM Pipeline (Raw BOLD Input)
%
% This script processes raw resting-state fMRI data (func_run1_df.nii)
% through the full SPM preprocessing pipeline:
%   1. Realignment
%   2. Slice timing correction
%   3. Coregistration (T1 to BOLD)
%   4. Segmentation (T1)
%   5. Normalization to MNI 2mm space (91x109x91)
%   6. Brain mask creation
%   7. BAT estimation (cerebellum surrogate)
%   8. ROI correlation map generation (136 channels)
%   9. DLRS_input feature map creation
%
% Required files in data/<subject>/PreprocessedData/:
%   - func_run1_df.nii   (raw defaced BOLD, native space)
%   - struct_rf.nii      (T1 structural)
%
% Required files in data/<subject>/:
%   - parameter_RS.txt   (pipeline parameters)
%   - slice_order_RS.txt (slice acquisition order)

clc;
clear all;
close all;

%% Navigate to code folder and data folder
[code_directory, ~, ~] = fileparts(mfilename('fullpath'));
addpath(code_directory);
addpath([code_directory, filesep, 'utils']);
addpath(genpath([code_directory, filesep, 'spm12']));

[parent_directory, ~, ~] = fileparts(code_directory);
data_directory = [parent_directory, filesep, 'data'];

%% Find subjects with raw BOLD data
subfolder = dir(data_directory);
subfolder(ismember({subfolder.name}, {'.', '..', '.gitkeeper', 'output', '.DS_Store'})) = [];
subfolder = subfolder([subfolder.isdir]);

valid_subjects = {};
for jj = 1:length(subfolder)
    if startsWith(subfolder(jj).name, 'sub-')
        subject_dir = [data_directory, filesep, subfolder(jj).name];
        preproc_dir = [subject_dir, filesep, 'PreprocessedData'];

        % Check for raw BOLD and structural files
        has_bold = exist([preproc_dir, filesep, 'func_run1_df.nii'], 'file');
        has_struct = exist([preproc_dir, filesep, 'struct_rf.nii'], 'file');
        has_params = exist([subject_dir, filesep, 'parameter_RS.txt'], 'file');

        if has_bold && has_struct && has_params
            % Check that this is NOT configured for AFNI mode
            fid = fopen([subject_dir, filesep, 'parameter_RS.txt'], 'r');
            content = fread(fid, '*char')';
            fclose(fid);

            if isempty(regexp(content, 'afniPreprocessed\s*:\s*1', 'once'))
                valid_subjects{end+1} = subfolder(jj).name;
            else
                fprintf('  Skipping %s (configured for AFNI mode)\n', subfolder(jj).name);
            end
        end
    end
end

if isempty(valid_subjects)
    fprintf('\nWARNING: No subjects found with raw BOLD data!\n');
    fprintf('Expected layout:\n');
    fprintf('  data/<subject>/PreprocessedData/func_run1_df.nii\n');
    fprintf('  data/<subject>/PreprocessedData/struct_rf.nii\n');
    fprintf('  data/<subject>/parameter_RS.txt\n');
    fprintf('  data/<subject>/slice_order_RS.txt\n');
    return;
end

fprintf('\n============================================================\n');
fprintf('DLRS_CVR_BAT - Original SPM Pipeline\n');
fprintf('============================================================\n\n');
fprintf('Found %d subject(s) with raw BOLD data:\n', length(valid_subjects));
for i = 1:length(valid_subjects)
    fprintf('  %d. %s\n', i, valid_subjects{i});
end
fprintf('\n');

%% Processing pipeline
for ii = 1:length(valid_subjects)
    fprintf('\n============================================================\n');
    fprintf('Processing subject %d/%d: %s\n', ii, length(valid_subjects), valid_subjects{ii});
    fprintf('============================================================\n\n');

    try
        % Step 1: Full SPM preprocessing (realign, slice timing, coregister, segment, normalize, mask)
        fprintf('[1/3] Running SPM preprocessing...\n');
        fprintf('  This includes: realignment, slice timing, coregistration,\n');
        fprintf('  segmentation, normalization, and brain mask creation.\n');
        fprintf('  This may take 15-30 minutes per subject.\n\n');
        RS_preprocessing(data_directory, valid_subjects{ii}, code_directory);

        % Step 2: BAT estimation using cerebellum surrogate
        fprintf('\n[2/3] Running BAT estimation...\n');
        RS_BAT(data_directory, valid_subjects{ii}, code_directory);

        % Step 3: Correlation map generation (136 channels)
        fprintf('\n[3/3] Generating CVR correlation maps...\n');
        RS_CVR_corrMap(data_directory, valid_subjects{ii}, code_directory);

        fprintf('\nSUBJECT COMPLETE: %s\n', valid_subjects{ii});

    catch ME
        fprintf('\nERROR processing %s:\n', valid_subjects{ii});
        fprintf('  Message: %s\n', ME.message);
        for kk = 1:length(ME.stack)
            fprintf('  Stack[%d]: %s (line %d)\n', kk, ME.stack(kk).name, ME.stack(kk).line);
        end
        continue;
    end
end

fprintf('\n============================================================\n');
fprintf('MATLAB PROCESSING COMPLETE\n');
fprintf('============================================================\n\n');
fprintf('Next steps:\n');
fprintf('  1. Verify DLRS_input files exist:\n');
fprintf('     ls data/*/DLRS_input/ | wc -l  (should show ~91 files per subject)\n\n');
fprintf('  2. Run Python inference:\n');
fprintf('     python src/DLRS_CVR_BAT_inference.py\n\n');
fprintf('  3. Visualize results:\n');
fprintf('     python visualize_results.py\n\n');
