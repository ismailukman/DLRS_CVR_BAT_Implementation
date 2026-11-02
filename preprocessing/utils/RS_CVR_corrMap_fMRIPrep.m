function RS_CVR_corrMap_fMRIPrep(wkdir, subname, code_directory)
% RS_CVR_corrMap_fMRIPrep
%
% CVR feature step for fMRIPrep input. Same as RS_CVR_corrMap_AFNI except:
%   - the CVR design is [CO2 surrogate, motion, motion.^2, constant], as in the
%     original RS_CVR_corrMap and Hou et al. (2023); motion is fMRIPrep's six
%     parameters, detrended and filtered at 0.1164 Hz like the BOLD
%   - censored volumes (censor_fmriprep.txt) are left out of every regression
%     and of the ROI correlation step; filtering uses the full series
%
% Closely follows the original RS_CVR_corrMap logic:
%   1. Load Neuromorphometrics atlas (91x109x91 - matches resampled data)
%   2. Load detrended BOLD, smooth, filter
%   3. Extract cerebellum CO2 surrogate
%   4. Compute CVR beta maps via GLM
%   5. Compute 134 ROI correlation maps
%   6. Combine into 136-channel features + BAT = 137 channels
%   7. Save as 2D DLRS_input layer files using img3Dto2D
%
% Usage: RS_CVR_corrMap_AFNI(data_dir, subject_name, code_dir)

fprintf('\n========================================\n');
fprintf('RS_CVR_corrMap_fMRIPrep - Generate features\n');
fprintf('========================================\n\n');

warning off;

%% Global Parameter Setup
mni_resolution = [2, 2, 2];
mni_dimension = [91, 109, 91];
mni_type = 16;

addpath([code_directory, filesep, 'utils']);
addpath([code_directory, filesep, 'spm12']);

cutfreq = 0.1164;

%% Load Neuromorphometric template (already 91x109x91)
mask_mni = spm_read_vols(spm_vol([code_directory, filesep, 'atlas', filesep, 'labels_Neuromorphometrics', filesep, 'wlabels_Neuromorphometrics_unique.nii']));
mask_mni_sub = mask_mni;
mask_list_eff = unique(mask_mni);
mask_list_eff = mask_list_eff(2:end);
mask_mni_len = length(mask_list_eff);

fprintf('Neuromorphometrics ROIs: %d\n\n', mask_mni_len);

%% Read parameters
cd([wkdir, filesep, subname]);
para_file_ID = fopen('parameter_RS.txt', 'r');
SmoothFWHMmm = 8;  % Default
TR = 2.0;

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
end
fclose(para_file_ID);

% Use 8mm smoothing if parameter says 0 (original pipeline uses 8mm here)
if SmoothFWHMmm == 0
    SmoothFWHMmm = 8;
end

fprintf('Parameters: TR=%.2f, Smooth=%dmm\n', TR, SmoothFWHMmm);

fs = 1/TR;
fcutoff = cutfreq / (fs/2);
[b, a] = butter(2, fcutoff);

bold_file_dir = 'PreprocessedData_RS';
bold_file_name = 'warfunc_run_combined';

% Load any volume to get dynnum
V_tmp = spm_vol([wkdir, filesep, subname, filesep, bold_file_dir, filesep, bold_file_name '.nii']);
dynnum = length(V_tmp);

%% Motion covariates and censor mask (from fmriprep/06_prepare_dlrs_input.py)
rp_data = load([wkdir, filesep, subname, filesep, 'rp_fmriprep.txt']);
keep = logical(load([wkdir, filesep, subname, filesep, 'censor_fmriprep.txt']));
if size(rp_data, 1) ~= dynnum || numel(keep) ~= dynnum
    error('Motion (%d) or censor (%d) length does not match BOLD (%d)', size(rp_data,1), numel(keep), dynnum);
end
for ii = 1:size(rp_data, 2)
    rp_data(:, ii) = filtfilt(b, a, detrend(rp_data(:, ii), 1));
end
fprintf('Motion covariates: 6 + squares; censored volumes: %d of %d\n', sum(~keep), dynnum);

%% Smooth the detrended BOLD images
fprintf('Smoothing detrended volumes (FWHM=%dmm)...\n', SmoothFWHMmm);
cd([wkdir, filesep, subname, filesep, bold_file_dir]);

P = cell(1,1);
P{1} = spm_select('List', [wkdir, filesep, subname, filesep, bold_file_dir], 'img', ['^dr' bold_file_name]);

if isempty(P{1})
    error('No detrended files found (dr%s*.img). Run RS_BAT_AFNI first.', bold_file_name);
end

V = spm_vol(P);
V = cat(1, V{:});

for ii = 1:length(V)
    [pth, nam, ext] = fileparts(V(ii).fname);
    fnameIn = fullfile(pth, [nam ext]);
    fname = fullfile(pth, ['s' int2str(SmoothFWHMmm) nam ext]);
    spm_smooth(fnameIn, fname, SmoothFWHMmm);
end
fprintf('  Done\n\n');

%% Load smoothed images
clear V;
smoothFiles = spm_vol(spm_select('List', [wkdir, filesep, subname, filesep, bold_file_dir], 'img', ['^s' int2str(SmoothFWHMmm) 'dr' bold_file_name]));

boldImg = zeros(mni_dimension(1), mni_dimension(2), mni_dimension(3), dynnum);
for ii = 1:length(smoothFiles)
    boldImg(:,:,:,ii) = spm_read_vols(spm_vol(smoothFiles(ii).fname));
end

boldImg_2D = reshape(boldImg, mni_dimension(1)*mni_dimension(2)*mni_dimension(3), dynnum);

%% Calculate the CVR map
fprintf('Computing CVR correlation maps...\n');

cerebellumImg = spm_read_vols(spm_vol([wkdir, filesep, subname, filesep, 'mask', filesep, 'brainMask_RS_cerebellum.nii']));
cerebellum_mask_loc = find(cerebellumImg == 1);

maskImg = spm_read_vols(spm_vol([wkdir, filesep, subname, filesep, 'mask', filesep, 'brainMask_RS.nii']));
mask_loc = find(maskImg == 1);

mask_mni_sub = maskImg .* mask_mni_sub;  % outside the brain = 0

% filter on the within-brain voxels
boldImg_2D_eff = boldImg_2D(mask_loc, :);
boldImg_2D_eff_filt = zeros(size(boldImg_2D_eff));
for ii = 1:size(boldImg_2D_eff, 1)
    if ~any(~isfinite(boldImg_2D_eff(ii, :)))
        boldImg_2D_eff_filt(ii, :) = filtfilt(b, a, boldImg_2D_eff(ii, :));
    end
end

boldImg_3D_filt = zeros(mni_dimension(1)*mni_dimension(2)*mni_dimension(3), 1);
boldImg_4D_filt = zeros(mni_dimension(1)*mni_dimension(2)*mni_dimension(3), dynnum);
for ii = 1:size(boldImg_2D_eff_filt, 2)
    boldImg_3D_filt(mask_loc) = boldImg_2D_eff_filt(:, ii);
    boldImg_4D_filt(:, ii) = boldImg_3D_filt;
end

% Using the averaged cerebellum signal as the surrogate for EtCO2
bestEtCO2 = mean(boldImg_4D_filt(cerebellum_mask_loc, :), 1)';
clear boldImg_4D_filt;

% Censored volumes are dropped from here on (after filtering the full series)
boldImg_2D_eff_filt = boldImg_2D_eff_filt(:, keep);
bestEtCO2 = bestEtCO2(keep);
rp_keep = rp_data(keep, :);
nkeep = sum(keep);

% CVR_beta0 = Mean bold image
CVR_beta0 = zeros(size(boldImg, 1)*size(boldImg, 2)*size(boldImg, 3), 1);
CVR_beta0(mask_loc) = mean(boldImg_2D_eff_filt, 2);

% Calculate CVR_beta1 with the original design [CO2, motion, motion.^2, constant]
CVR_beta1 = zeros(size(boldImg, 1)*size(boldImg, 2)*size(boldImg, 3), 1);

regression_x = cat(2, bestEtCO2, rp_keep, rp_keep.^2, ones(nkeep, 1));
regrssion_x_psu = (regression_x'*regression_x)\regression_x';
regression_para = regrssion_x_psu * boldImg_2D_eff_filt';
CVR_beta1(mask_loc) = regression_para(1, :);

% Regress out the CO2 signal and motion (as the original RS_CVR_corrMap)
boldImg_2D_eff_filt = boldImg_2D_eff_filt - (regression_x(:, 1:end-1)*regression_para(1:end-1, :))';
boldImg_2D_eff_filt = zscore(boldImg_2D_eff_filt, 0, 2);

boldImg_3D_filt_res = zeros(mni_dimension(1)*mni_dimension(2)*mni_dimension(3), 1);
boldImg_4D_filt_res = zeros(mni_dimension(1)*mni_dimension(2)*mni_dimension(3), nkeep);
for ii = 1:size(boldImg_2D_eff_filt, 2)
    boldImg_3D_filt_res(mask_loc) = boldImg_2D_eff_filt(:, ii);
    boldImg_4D_filt_res(:, ii) = boldImg_3D_filt_res;
end

% Average BOLD signal in each ROI mask
boldImg_roi_1D_res = zeros(mask_mni_len, nkeep);
for ii = 1:mask_mni_len
    boldImg_roi_2D = boldImg_4D_filt_res(mask_mni_sub == mask_list_eff(ii), :);
    boldImg_roi_1D_res(ii, :) = mean(boldImg_roi_2D, 1);
    clear boldImg_roi_2D;
end

fprintf('Computing ROI correlation maps (%d ROIs)...\n', mask_mni_len);

% Calculate correlation maps
corrMap = zeros(mni_dimension(1)*mni_dimension(2)*mni_dimension(3), mask_mni_len);
for uu = 1:mask_mni_len
    if mod(uu, 20) == 0
        fprintf('  ROI %d/%d\n', uu, mask_mni_len);
    end
    
    bestEtCO2_roi_res = boldImg_roi_1D_res(uu, :)';
    bestEtCO2_roi_res = zscore(bestEtCO2_roi_res, 0, 1);
    
    regression_x_roi = cat(2, bestEtCO2_roi_res, ones(nkeep, 1));
    CVR_roi = zeros(size(boldImg, 1)*size(boldImg, 2)*size(boldImg, 3), 1);
    
    regrssion_x_roi_psu = (regression_x_roi'*regression_x_roi)\regression_x_roi';
    beta_roi_para = regrssion_x_roi_psu * boldImg_2D_eff_filt';
    
    CVR_roi(mask_loc) = beta_roi_para(1, :)';
    corrMap(:, uu) = CVR_roi;
    clear CVR_roi;
end

fprintf('  Done\n\n');

%% Combine CVR beta maps and corrMaps (134 + 2 = 136 channels)
CVR_4D = reshape(cat(2, corrMap, CVR_beta0, CVR_beta1), mni_dimension(1), mni_dimension(2), mni_dimension(3), size(corrMap, 2)+2);

%% Load BAT map from CVR_mapping step
fprintf('Loading BAT map...\n');

% The original pipeline produces *_s8_co2delay.img
batFile = dir([wkdir, filesep, subname, filesep, 'CVR_voxelshift_etco2_cerebellum', filesep, '*_s*_co2delay.img']);
if isempty(batFile)
    % Try without smoothing prefix
    batFile = dir([wkdir, filesep, subname, filesep, 'CVR_voxelshift_etco2_cerebellum', filesep, '*_co2delay.img']);
end
if isempty(batFile)
    error('BAT file (*_co2delay.img) not found. Run RS_BAT_AFNI first.');
end
batImg = spm_read_vols(spm_vol([batFile(1).folder, filesep, batFile(1).name]));

% Add BAT as channel 137
CVR_4D = cat(4, CVR_4D, batImg);

fprintf('Feature tensor: %d x %d x %d x %d channels\n', size(CVR_4D));
fprintf('  (134 ROI correlations + CVR_beta0 + CVR_beta1 + BAT)\n\n');

%% Normalize each channel (z-score within brain mask)
for uu = 1:size(CVR_4D, 4)
    CVR_4D(:,:,:,uu) = rewrite3D(CVR_4D(:,:,:,uu), maskImg);
end

%% Save as 2D DLRS_input layer files using original img3Dto2D
fprintf('Saving DLRS_input feature maps...\n');

mkdir([wkdir, filesep, subname, filesep, 'DLRS_input']);
prefix_mri = [wkdir, filesep, subname, filesep, 'DLRS_input', filesep, 'DLRS_input_layer_'];
img3Dto2D(CVR_4D, prefix_mri, 1, 0, 0);

fprintf('  Saved to: %s\n', [wkdir, filesep, subname, filesep, 'DLRS_input']);
fprintf('  Number of layers: 91\n');
fprintf('  Channels per layer: %d\n\n', size(CVR_4D, 4));

fprintf('========================================\n');
fprintf('RS_CVR_corrMap_AFNI COMPLETE\n');
fprintf('========================================\n\n');

end

%% Helper function (same as original RS_CVR_corrMap)
function img_f = rewrite3D(img, mask)

img(isnan(img)) = 0;

mask_loc = find(mask == 1);
mask_loc0 = find(mask == 0);
img(mask_loc0) = nan;
img_mean = mean(img(mask_loc));
img_std = std(img(mask_loc), 0, 1);
img_re = (img - img_mean) / img_std;
img_f = img_re;
img_f(mask_loc0) = 0;

end
