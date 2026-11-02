function RS_preprocessing_AFNI(wkdir, folder_name, code_directory)
% RS_preprocessing_AFNI
%
% Prepares already preprocessed BOLD data for the BAT and CVR steps. Used for
% both AFNI errts input and fMRIPrep input (parameter_RS.txt: fmriprepInput 1).
% Realignment, slice timing, coregistration, normalization and segmentation
% are skipped because they were done upstream.
%
% Steps performed:
%   1. Use func_run_combined.nii (fMRIPrep, written by
%      fmriprep/06_prepare_dlrs_input.py) or concatenate the errts runs (AFNI)
%   2. Resample to the 91x109x91 2 mm MNI152 grid of the pretrained model
%   3. Create the brain mask and the cerebellum mask; for fMRIPrep input,
%      cerebellar voxels with little signal are removed from the mask
%
% Usage: RS_preprocessing_AFNI(data_dir, subject_name, code_dir)

fprintf('\n========================================\n');
fprintf('RS_PREPROCESSING_AFNI - AFNI Input Mode\n');
fprintf('========================================\n\n');

cd([wkdir, filesep, folder_name]);
para_file_ID = fopen('parameter_RS.txt', 'r');

% Read parameters
SmoothFWHMmm = 0;
TR = 2.0;
fmriprep_input = 0;

while ~feof(para_file_ID)
    tline = fgetl(para_file_ID);
    if regexp(tline, 'SmoothFWHM')
        colon_loc = regexp(tline, ':');
        SmoothFWHMmm = str2num(tline(colon_loc+1:end));
    end
    if regexp(tline, 'TRs')
        colon_loc = regexp(tline, ':');
        TR = str2num(tline(colon_loc+1:end));
    end
    if regexp(tline, 'fmriprepInput')
        colon_loc = regexp(tline, ':');
        fmriprep_input = str2num(tline(colon_loc+1:end));
    end
end
fclose(para_file_ID);

fprintf('Parameters loaded:\n');
fprintf('  TR: %.2f s\n', TR);
fprintf('  Additional smoothing: %.1f mm\n\n', SmoothFWHMmm);

%% Global Parameter Setup
addpath([code_directory, filesep, 'utils']);
addpath([code_directory, filesep, 'spm12']);

spm_get_defaults;
global defaults;
defaults.mask.thresh = 0;

mni_resolution = [2, 2, 2];
mni_dimension = [91, 109, 91];
mni_type = 16;

%% STEP 1: Find and load AFNI errts files
fprintf('========================================\n');
fprintf('STEP 1: Find and load AFNI errts files\n');
fprintf('========================================\n\n');

subj_root = [wkdir, filesep, folder_name];
rs_dir = [subj_root, filesep, 'PreprocessedData_RS'];
if ~exist(rs_dir, 'dir')
    mkdir(rs_dir);
end

bold_file_name = 'func_run_combined';

% Check if we already have the final resampled file
war_file = [rs_dir, filesep, 'war' bold_file_name '.nii'];
if exist(war_file, 'file')
    fprintf('Using existing resampled file: %s\n\n', war_file);
else
    % Check if already concatenated (native space)
    combined_file = [subj_root, filesep, bold_file_name '.nii'];
    
    if exist(combined_file, 'file')
        fprintf('Using pre-concatenated file: %s\n', combined_file);
    else
        % Search for AFNI errts files in subject root
        fprintf('Searching for AFNI errts files in subject root...\n');
        
        % Try .nii first, then .nii.gz
        pattern_r01 = dir([subj_root, filesep, 'errts.*.r01.*fanaticor*.nii']);
        pattern_r02 = dir([subj_root, filesep, 'errts.*.r02.*fanaticor*.nii']);
        
        % Filter out .nii.gz matches (dir '*.nii' also matches '*.nii.gz')
        if ~isempty(pattern_r01)
            pattern_r01 = pattern_r01(~endsWith({pattern_r01.name}, '.gz'));
        end
        if ~isempty(pattern_r02)
            pattern_r02 = pattern_r02(~endsWith({pattern_r02.name}, '.gz'));
        end
        
        % If no uncompressed .nii, try .nii.gz and decompress
        if isempty(pattern_r01) || isempty(pattern_r02)
            pattern_r01_gz = dir([subj_root, filesep, 'errts.*.r01.*fanaticor*.nii.gz']);
            pattern_r02_gz = dir([subj_root, filesep, 'errts.*.r02.*fanaticor*.nii.gz']);
            
            if ~isempty(pattern_r01_gz) && ~isempty(pattern_r02_gz)
                fprintf('  Decompressing .nii.gz files...\n');
                gunzip(fullfile(pattern_r01_gz(1).folder, pattern_r01_gz(1).name));
                gunzip(fullfile(pattern_r02_gz(1).folder, pattern_r02_gz(1).name));
                
                pattern_r01 = dir([subj_root, filesep, 'errts.*.r01.*fanaticor*.nii']);
                pattern_r02 = dir([subj_root, filesep, 'errts.*.r02.*fanaticor*.nii']);
                pattern_r01 = pattern_r01(~endsWith({pattern_r01.name}, '.gz'));
                pattern_r02 = pattern_r02(~endsWith({pattern_r02.name}, '.gz'));
            end
        end
        
        if isempty(pattern_r01)
            error('Could not find AFNI errts fanaticor r01 file in: %s', subj_root);
        end

        errts_r01 = fullfile(pattern_r01(1).folder, pattern_r01(1).name);
        fprintf('  Found R01: %s\n', pattern_r01(1).name);
        V_r01 = spm_vol(errts_r01);
        fprintf('  Loading R01 (%d volumes)...\n', length(V_r01));
        data_r01 = spm_read_vols(V_r01);

        if ~isempty(pattern_r02)
            % Two runs present: concatenate along the time dimension (default)
            errts_r02 = fullfile(pattern_r02(1).folder, pattern_r02(1).name);
            fprintf('  Found R02: %s\n', pattern_r02(1).name);
            V_r02 = spm_vol(errts_r02);
            fprintf('  Loading R02 (%d volumes)...\n', length(V_r02));
            data_r02 = spm_read_vols(V_r02);
            bold_data = cat(4, data_r01, data_r02);
            clear data_r01 data_r02;
            fprintf('  Concatenated to %d volumes\n', size(bold_data, 4));
        else
            % Single run only (used for test-retest per-run processing)
            bold_data = data_r01;
            clear data_r01;
            fprintf('  Single run: %d volumes\n', size(bold_data, 4));
        end
        
        % Save concatenated file to subject root
        fprintf('\nSaving concatenated file: %s\n', combined_file);
        V_out = V_r01(1);
        V_out.fname = combined_file;
        for vol = 1:size(bold_data, 4)
            V_out.n = [vol, 1];
            spm_write_vol(V_out, bold_data(:,:,:,vol));
        end
        fprintf('  Saved\n');
        clear bold_data;
    end
    
    %% STEP 2: Resample to 91x109x91 MNI152 (2mm)
    fprintf('\n========================================\n');
    fprintf('STEP 2: Resample to MNI152 (91x109x91)\n');
    fprintf('========================================\n\n');
    
    fprintf('AFNI data is 64x76x64 at 3mm (MNI_2009c_asym)\n');
    fprintf('Model expects 91x109x91 at 2mm (MNI152)\n');
    fprintf('Resampling...\n\n');
    
    % Use the Neuromorphometrics atlas as reference for target space
    atlas_path = [code_directory, filesep, 'atlas', filesep, 'labels_Neuromorphometrics', filesep, 'wlabels_Neuromorphometrics_unique.nii'];
    V_ref = spm_vol(atlas_path);
    
    V_src = spm_vol(combined_file);
    nVol = length(V_src);
    
    % Create output volume template matching atlas space
    V_out = V_ref;
    V_out.fname = war_file;
    V_out.dt = [16, 0];
    
    % Pre-compute mapping from target to source space
    M_target = V_ref.mat;
    M_source = V_src(1).mat;
    M = M_source \ M_target;  % target-to-source transformation
    
    [X, Y, Z] = ndgrid(1:mni_dimension(1), 1:mni_dimension(2), 1:mni_dimension(3));
    coords = [X(:)'; Y(:)'; Z(:)'; ones(1, numel(X))];
    src_coords = M * coords;
    
    fprintf('  Resampling %d volumes...\n', nVol);
    for vol = 1:nVol
        if mod(vol, 50) == 0
            fprintf('    Volume %d/%d\n', vol, nVol);
        end
        V_out.n = [vol, 1];
        data_vol = spm_read_vols(V_src(vol));
        
        % Interpolate
        resampled = interp3(data_vol, src_coords(2,:), src_coords(1,:), src_coords(3,:), 'linear', 0);
        resampled = reshape(resampled, mni_dimension);
        
        spm_write_vol(V_out, resampled);
    end
    fprintf('  Resampled file saved: %s\n\n', war_file);
end

%% STEP 3: Create brain mask from Neuromorphometrics atlas (proper brain shape)
fprintf('========================================\n');
fprintf('STEP 3: Create brain mask (atlas-based)\n');
fprintf('========================================\n\n');

mask_dir = [subj_root, filesep, 'mask'];
if ~exist(mask_dir, 'dir')
    mkdir(mask_dir);
end

brainMaskFile = [mask_dir, filesep, 'brainMask_RS.nii'];

% Always recreate mask using atlas to ensure proper brain shape
fprintf('Creating brain mask from Neuromorphometrics atlas...\n');

% Use atlas as brain template (all labeled voxels = brain)
atlas_path = [code_directory, filesep, 'atlas', filesep, 'labels_Neuromorphometrics', filesep, 'wlabels_Neuromorphometrics_unique.nii'];
atlas_data = spm_read_vols(spm_vol(atlas_path));
brainMask = double(atlas_data > 0);

% Morphological cleanup: close small gaps, fill holes
se = strel('sphere', 2);
brainMask = imclose(brainMask, se);
brainMask = imfill(brainMask, 'holes');

% Intersect with BOLD signal FOV (keep only voxels that have data)
% NOTE: AFNI errts data are regression residuals with mean ~ 0,
% so we use temporal std > 0 to detect valid voxels (not mean).
V_war = spm_vol(war_file);
nVol = length(V_war);

% Compute temporal standard deviation to find voxels with actual signal
sum_bold = zeros(mni_dimension);
sum_bold2 = zeros(mni_dimension);
for vol = 1:nVol
    vol_data = spm_read_vols(V_war(vol));
    sum_bold = sum_bold + vol_data;
    sum_bold2 = sum_bold2 + vol_data.^2;
end
mean_bold = sum_bold / nVol;
std_bold = sqrt(sum_bold2/nVol - mean_bold.^2);

% Voxels inside FOV have non-zero temporal variance; outside FOV are exactly 0
bold_fov = double(std_bold > 1e-6);
brainMask = brainMask .* bold_fov;
fprintf('  BOLD FOV voxels: %d\n', sum(bold_fov(:)));

write_hdrimg(brainMask, brainMaskFile, mni_resolution, mni_type);
fprintf('  Atlas-based brain mask saved: %s\n', brainMaskFile);

brainVox = find(brainMask == 1);
fprintf('  Brain voxels: %d\n\n', length(brainVox));

%% STEP 4: Create cerebellum mask
fprintf('========================================\n');
fprintf('STEP 4: Create cerebellum mask\n');
fprintf('========================================\n\n');

cerMaskFile = [mask_dir, filesep, 'brainMask_RS_cerebellum.nii'];

% Always recreate cerebellum mask to match updated brain mask
% Cerebellum labels for wlabels_Neuromorphometrics_unique.nii, which is
% renumbered 1..133. The standard SPM atlas numbers (38 39 40 41 71 72 73)
% point at frontal and insular regions in this file, not cerebellum.
% Mapping verified by resampling the standard atlas and matching labels by
% maximum voxel overlap (93-98% per label).
cer_labels = [10 11 12 13 31 32 33];
cerMask = double(ismember(round(atlas_data), cer_labels));
cerMask(brainMask == 0) = 0;

% fMRIPrep output is not masked: where the scan does not reach the lower
% cerebellum, voxels hold little or no signal. Keep only cerebellar voxels
% whose mean signal is at least 20% of the cerebellar median, so the
% reference signal is not diluted by empty voxels.
if fmriprep_input
    n_before = sum(cerMask(:));
    cer_median = median(mean_bold(cerMask == 1));
    cerMask(mean_bold < 0.2 * cer_median) = 0;
    fprintf('  Cerebellar voxels with signal: %d of %d (%.0f%%)\n', ...
            sum(cerMask(:)), n_before, 100 * sum(cerMask(:)) / n_before);
end

write_hdrimg(cerMask, cerMaskFile, mni_resolution, mni_type);
fprintf('  Cerebellum mask saved (labels: %s)\n', num2str(cer_labels));

cerVox = find(cerMask == 1);
fprintf('Cerebellum voxels: %d\n', length(cerVox));

if length(cerVox) == 0
    error('No cerebellum voxels found! Check atlas and mask.');
end

% guard against a wrong label set: the mask must sit in the posterior, inferior brain
V_atlas = spm_vol(atlas_path);
[ci, cj, ck] = ind2sub(size(cerMask), cerVox);
cer_mni = V_atlas.mat * [ci'; cj'; ck'; ones(1, numel(ci))];
if mean(cer_mni(2, :)) > -40 || mean(cer_mni(3, :)) > -20
    error('Cerebellum mask centroid at y=%.0f, z=%.0f mm is not in the cerebellum; check cer_labels.', ...
          mean(cer_mni(2, :)), mean(cer_mni(3, :)));
end
fprintf('  Cerebellum mask centroid: x=%.0f y=%.0f z=%.0f mm (MNI)\n', mean(cer_mni(1:3, :), 2));

fprintf('\n========================================\n');
fprintf('RS_PREPROCESSING_AFNI COMPLETE\n');
fprintf('========================================\n\n');

end
