%% DLRS_CVR_BAT - MATLAB feature generation for all participants, fMRIPrep input
%
% Processes the per-participant folders written by
% fmriprep/06_prepare_dlrs_input.py (func_run_combined.nii, rp_fmriprep.txt,
% censor_fmriprep.txt). Motion covariates and 0.4 censoring are applied in
% RS_BAT_fMRIPrep and RS_CVR_corrMap_fMRIPrep.
%
% Usage in MATLAB:
%   cd preprocessing
%   rs_running_fmriprep
%   exit

clc;
close all;

%% Setup paths
[code_directory, ~, ~] = fileparts(mfilename('fullpath'));
addpath(code_directory);
addpath([code_directory, filesep, 'utils']);
addpath(genpath([code_directory, filesep, 'spm12']));

data_directory = getenv('DLRS_MATLAB_DATA');
if isempty(data_directory)
    data_directory = '/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/04_DERIVATIVES/02_DLRS_INPUT/fmriprep';
end
% each participant is processed in a local folder and only the results are copied
% back, so the external drive is not written to for hours
local_root = fullfile(getenv('HOME'), 'dlrs_tmp');
if exist('only_subjects', 'var') == 0
    only_subjects = {};   % e.g. {'sci02'} to process a subset
end

group_names = {'HC_subj', 'SCI_subj'};

%% Discover subjects in each group
all_jobs = {};  % Each entry: {group_data_dir, subject_id, group_name}

for gg = 1:length(group_names)
    group_dir = [data_directory, filesep, group_names{gg}];
    if ~exist(group_dir, 'dir')
        fprintf('WARNING: Group directory not found: %s\n', group_dir);
        continue;
    end
    
    subfolder = dir(group_dir);
    subfolder(ismember({subfolder.name}, {'.', '..', '.DS_Store'})) = [];
    subfolder = subfolder([subfolder.isdir]);
    
    for jj = 1:length(subfolder)
        subj_dir = [group_dir, filesep, subfolder(jj).name];
        
        if ~isempty(only_subjects) && ~ismember(subfolder(jj).name, only_subjects)
            continue;
        end
        has_combined = exist([subj_dir, filesep, 'func_run_combined.nii'], 'file') && ...
                       exist([subj_dir, filesep, 'rp_fmriprep.txt'], 'file');

        % Skip subjects already fully processed (DLRS_input has 91 layers).
        % Lets the run resume after interruption and avoids reprocessing the
        % already-done subjects when the cohort is extended.
        n_layers = numel(dir([subj_dir, filesep, 'DLRS_input', filesep, ...
                              'DLRS_input_layer_*.nii']));
        if n_layers >= 91
            fprintf('  SKIP (already done, %d layers): [%s] %s\n', ...
                    n_layers, group_names{gg}, subfolder(jj).name);
            continue;
        end

        if has_combined
            all_jobs{end+1} = {group_dir, subfolder(jj).name, group_names{gg}};
        end
    end
end

if isempty(all_jobs)
    fprintf('\nERROR: No subjects found!\n');
    fprintf('Run fmriprep/06_prepare_dlrs_input.py first.\n\n');
    return;
end

fprintf('\n============================================================\n');
fprintf('DLRS_CVR_BAT - Group Processing (HC vs SCI), fMRIPrep input\n');
fprintf('============================================================\n\n');
fprintf('Total subjects to process: %d\n\n', length(all_jobs));
for ii = 1:length(all_jobs)
    fprintf('  %2d. [%s] %s\n', ii, all_jobs{ii}{3}, all_jobs{ii}{2});
end
fprintf('\n');

%% Process each subject
success_count = 0;
fail_count = 0;

for ii = 1:length(all_jobs)
    group_data_dir = all_jobs{ii}{1};
    subj_name = all_jobs{ii}{2};
    group_name = all_jobs{ii}{3};
    
    fprintf('\n============================================================\n');
    fprintf('Processing %d/%d: [%s] %s\n', ii, length(all_jobs), group_name, subj_name);
    fprintf('============================================================\n\n');
    
    if ~exist(group_data_dir, 'dir')
        fprintf('\nData folder not reachable (drive disconnected?): %s\nStopping.\n', group_data_dir);
        break;
    end
    local_group = fullfile(local_root, group_name);
    local_subj = fullfile(local_group, subj_name);
    if exist(local_subj, 'dir'), rmdir(local_subj, 's'); end
    mkdir(local_subj);
    src = fullfile(group_data_dir, subj_name);
    for f = {'func_run_combined.nii', 'rp_fmriprep.txt', 'censor_fmriprep.txt', 'parameter_RS.txt', 'run_info.json'}
        copyfile(fullfile(src, f{1}), local_subj);
    end

    try
        % each step takes (group folder, participant label, code folder)
        
        fprintf('[1/3] Preprocessing...\n');
        RS_preprocessing_AFNI(local_group, subj_name, code_directory);
        
        fprintf('\n[2/3] BAT estimation...\n');
        RS_BAT_fMRIPrep(local_group, subj_name, code_directory);
        
        fprintf('\n[3/3] CVR correlation maps...\n');
        RS_CVR_corrMap_fMRIPrep(local_group, subj_name, code_directory);

        % copy results back to the drive, then remove the local copy
        for d = {'DLRS_input', 'CVR_voxelshift_etco2_cerebellum', 'mask'}
            if exist(fullfile(src, d{1}), 'dir'), rmdir(fullfile(src, d{1}), 's'); end
            copyfile(fullfile(local_subj, d{1}), fullfile(src, d{1}));
        end
        copyfile(fullfile(local_subj, 'avgBOLD_cerebellum.txt'), src);
        rmdir(local_subj, 's');
        
        fprintf('\nCOMPLETE: [%s] %s\n', group_name, subj_name);
        success_count = success_count + 1;
        
    catch ME
        fprintf('\nERROR: [%s] %s\n', group_name, subj_name);
        fprintf('  %s\n', ME.message);
        for kk = 1:min(3, length(ME.stack))
            fprintf('  Stack[%d]: %s (line %d)\n', kk, ME.stack(kk).name, ME.stack(kk).line);
        end
        fail_count = fail_count + 1;
        continue;
    end
end

fprintf('\n============================================================\n');
fprintf('MATLAB PROCESSING COMPLETE\n');
fprintf('============================================================\n');
fprintf('  Successful: %d/%d\n', success_count, length(all_jobs));
fprintf('  Failed:     %d/%d\n', fail_count, length(all_jobs));
fprintf('============================================================\n\n');
fprintf('Next steps:\n');
fprintf('  python src/DLRS_CVR_BAT_inference.py\n');
fprintf('  python src/analysis/regional_means.py\n\n');
