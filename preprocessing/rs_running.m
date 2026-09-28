%% DLRS_CVR_BAT - AFNI Preprocessed Input Mode
% 
% This script processes AFNI-preprocessed resting-state fMRI data
% using the DLRS_CVR_BAT pipeline
%
% Input files required:
%   - errts.*.r01.fanaticor+tlrc.nii.gz (AFNI preprocessed, run 1)
%   - errts.*.r02.fanaticor+tlrc.nii.gz (AFNI preprocessed, run 2)
%
% After AFNI_to_SPM_pipeline.sh conversion, this script will:
%   1. Extract cerebellum signal as CO2 surrogate
%   2. Calculate BAT (bolus arrival time) maps
%   3. Generate ROI correlation maps (136 channels)
%   4. Create DLRS_input feature maps for deep learning inference
%
% Usage in MATLAB:
%   cd preprocessing
%   rs_running
%   exit

clc;
clear all;
close all;

%% Navigate to code folder and data folder
[code_directory, ~, ~] = fileparts(mfilename('fullpath'));
addpath(code_directory);
addpath([code_directory, filesep, 'utils']);

[parent_directory, ~, ~] = fileparts(code_directory);
data_directory = [parent_directory, filesep, 'data'];

%% Find subjects with AFNI preprocessed data
subfolder = dir(data_directory);
subfolder(ismember({subfolder.name}, {'.', '..', '.gitkeeper', 'output', '.DS_Store'})) = [];
subfolder = subfolder([subfolder.isdir]);

% Find subjects (sub-*)
valid_subjects = {};
for jj = 1:length(subfolder)
    if startsWith(subfolder(jj).name, 'sub-')
        subject_dir = [data_directory, filesep, subfolder(jj).name];
        
        % Check for AFNI errts files (.nii or .nii.gz) or already concatenated file
        has_errts = ~isempty(dir([subject_dir, filesep, 'errts.*.r01.*fanaticor*.nii*']));
        has_combined = exist([subject_dir, filesep, 'func_run_combined.nii'], 'file');
        
        if has_errts || has_combined
            valid_subjects{end+1} = subfolder(jj).name;
            
            % Ensure parameter file exists with correct AFNI settings
            param_file = [subject_dir, filesep, 'parameter_RS.txt'];
            if ~exist(param_file, 'file')
                fprintf('  Creating parameter_RS.txt for %s\n', subfolder(jj).name);
                fid = fopen(param_file, 'w');
                fprintf(fid, '"UserInput" : {\n');
                fprintf(fid, '\t "SmoothFWHMmm" : 0\n');
                fprintf(fid, '\t "TRs" : 2.0\n');
                fprintf(fid, '\t "mprageFileName" : ""\n');
                fprintf(fid, '\t "boldFileName_RS" : ""\n');
                fprintf(fid, '\t "sliceOrderFile_RS" : ""\n');
                fprintf(fid, '\t "refSlice" : 1\n');
                fprintf(fid, '\t "afniPreprocessed" : 1\n');
                fprintf(fid, '}\n');
                fclose(fid);
            end
        end
    end
end

if isempty(valid_subjects)
    fprintf('\nWARNING: No AFNI preprocessed subjects found!\n');
    fprintf('Please run: bash preprocessing/AFNI_to_SPM_pipeline.sh <data_dir> <subject>\n');
    fprintf('Example: bash preprocessing/AFNI_to_SPM_pipeline.sh ./data sub-03a\n\n');
    return;
end

fprintf('\n============================================================\n');
fprintf('DLRS_CVR_BAT - AFNI Preprocessed Input Mode\n');
fprintf('============================================================\n\n');
fprintf('Found %d subject(s) with AFNI preprocessing:\n', length(valid_subjects));
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
        % Run preprocessing
        fprintf('[1/3] Running preprocessing...\n');
        RS_preprocessing_AFNI(data_directory, valid_subjects{ii}, code_directory);
        
        % Run BAT estimation
        fprintf('\n[2/3] Running BAT estimation...\n');
        RS_BAT_AFNI(data_directory, valid_subjects{ii}, code_directory);
        
        % Run correlation map generation
        fprintf('\n[3/3] Generating CVR correlation maps...\n');
        RS_CVR_corrMap_AFNI(data_directory, valid_subjects{ii}, code_directory);
        
        fprintf('\nSUBJECT COMPLETE: %s\n', valid_subjects{ii});
        
    catch ME
        fprintf('\nERROR processing %s:\n', valid_subjects{ii});
        fprintf('  %s\n', ME.message);
        fprintf('  Stack: %s\n', ME.stack(1).name);
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
