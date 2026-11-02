function RS_BAT_fMRIPrep(wkdir, subname, code_directory)
% RS_BAT_fMRIPrep
%
% BAT step for fMRIPrep input (prepared by fmriprep/06_prepare_dlrs_input.py
% and RS_preprocessing_AFNI, which only copies it onto the atlas grid).
%
% Same as RS_BAT_AFNI except:
%   - motion covariates are restored: fMRIPrep's six motion parameters
%     (rp_fmriprep.txt), filtered at 0.1164 Hz inside CVR_mapping_voxelwise_GLM_RS,
%     plus their squares inside cvr_func_glm_RS, as in Hou et al. (2023)
%   - censored volumes (censor_fmriprep.txt, motion above 0.4) are left out of
%     the delay-search GLM; the signal is filtered on the full series first
%
% Usage: RS_BAT_fMRIPrep(data_dir, subject_name, code_dir)

fprintf('\n========================================\n');
fprintf('RS_BAT_fMRIPrep - Calculate BAT maps\n');
fprintf('========================================\n\n');

cd([wkdir, filesep, subname]);
mni_resolution = [2, 2, 2];
mni_dimension = [91, 109, 91];
mni_type = 16;

addpath([code_directory, filesep, 'utils']);
addpath([code_directory, filesep, 'spm12']);

cutfreq = 0.1164;  % cutoff frequency

%% Read parameters
para_file_ID = fopen('parameter_RS.txt', 'r');
TR = 2.0;
SmoothFWHMmm = 8;  % Default smoothing for BAT (matches original)

while ~feof(para_file_ID)
    tline = fgetl(para_file_ID);
    if regexp(tline, 'TRs')
        colon_loc = regexp(tline, ':');
        TR = str2num(tline(colon_loc+1:end));
    end
    if regexp(tline, 'SmoothFWHM')
        colon_loc = regexp(tline, ':');
        SmoothFWHMmm = str2num(tline(colon_loc+1:end));
    end
end
fclose(para_file_ID);

% BAT calculation needs some smoothing - use 8mm if parameter says 0
if SmoothFWHMmm == 0
    SmoothFWHMmm = 8;
end

fprintf('Parameters: TR=%.2f, Smooth=%dmm\n\n', TR, SmoothFWHMmm);

fs = 1/TR;
fcutoff = cutfreq / (fs/2);
[b, a] = butter(2, fcutoff);

%% Load resampled BOLD (91x109x91)
bold_file_dir = 'PreprocessedData_RS';
bold_file_name = 'warfunc_run_combined';

bold_path = [wkdir, filesep, subname, filesep, bold_file_dir, filesep, bold_file_name '.nii'];
if ~exist(bold_path, 'file')
    error('Resampled BOLD not found: %s\nRun RS_preprocessing_AFNI first.', bold_path);
end

fprintf('Loading resampled BOLD: %s\n', bold_path);
V_bold = spm_vol(bold_path);
boldImg = spm_read_vols(V_bold);
dynnum = size(boldImg, 4);
fprintf('  Dimensions: %d x %d x %d x %d\n\n', size(boldImg));

%% Detrend
fprintf('Detrending...\n');
mask = spm_read_vols(spm_vol([wkdir, filesep, subname, filesep, 'mask', filesep, 'brainMask_RS.nii']));
brainVox = find(mask == 1);
mask_size = size(mask);
detrendImg_4D = zeros(size(boldImg));

for vox = 1:length(brainVox)
    [row, col, sl] = ind2sub(mask_size, brainVox(vox));
    TS1 = squeeze(boldImg(row, col, sl, :));
    
    if any(isnan(TS1)) || length(TS1) < 2
        continue;
    end
    
    sig = detrend(TS1);
    meansig = mean(TS1 - sig);
    TS2 = sig + meansig;
    detrendImg_4D(row, col, sl, :) = TS2;
end
fprintf('  Done\n\n');

%% Write detrended images as .img files (needed by CVR_mapping_voxelwise_GLM_RS)
fprintf('Writing detrended volumes as .img files...\n');
ftemp_dir = [wkdir, filesep, subname, filesep, bold_file_dir];
ftemp_name = ['dr' bold_file_name];

for ii = 1:size(detrendImg_4D, 4)
    if ii < 10
        write_hdrimg(detrendImg_4D(:,:,:,ii), [ftemp_dir, filesep, ftemp_name, '-00', num2str(ii), '-001.img'], mni_resolution, mni_type);
    elseif ii < 100
        write_hdrimg(detrendImg_4D(:,:,:,ii), [ftemp_dir, filesep, ftemp_name, '-0', num2str(ii), '-001.img'], mni_resolution, mni_type);
    else
        write_hdrimg(detrendImg_4D(:,:,:,ii), [ftemp_dir, filesep, ftemp_name, '-', num2str(ii), '-001.img'], mni_resolution, mni_type);
    end
end
fprintf('  Written %d volumes\n\n', size(detrendImg_4D, 4));

boldImg_2D = reshape(detrendImg_4D, mni_dimension(1)*mni_dimension(2)*mni_dimension(3), dynnum);

%% Cerebellum mask - extract CO2 surrogate
fprintf('Extracting cerebellum signal (CO2 surrogate)...\n');
grayImg = spm_read_vols(spm_vol([wkdir, filesep, subname, filesep, 'mask', filesep, 'brainMask_RS_cerebellum.nii']));
graymask_loc = find(grayImg == 1);
boldImg_4D_eff = boldImg_2D(graymask_loc, :);

% Filter
boldImg_4D_eff_filt = zeros(size(boldImg_4D_eff));
for ii = 1:size(boldImg_4D_eff, 1)
    if any(~isfinite(boldImg_4D_eff(ii, :)))
        boldImg_4D_eff_filt(ii, :) = nan;
    else
        boldImg_4D_eff_filt(ii, :) = filtfilt(b, a, boldImg_4D_eff(ii, :));
    end
end

bestEtCO2 = nanmean(boldImg_4D_eff_filt, 1)';
nave = round(length(bestEtCO2)/4);
[Y, I] = sort(bestEtCO2, 'descend');
EtCO2_min = mean(Y(end-nave:end));
EtCO2_mean = mean(Y);
EtCO2_max = mean(Y(1:nave));

fprintf('  CO2 signal stats: min=%.4f, mean=%.4f, max=%.4f\n\n', EtCO2_min, EtCO2_mean, EtCO2_max);

%% Save cerebellum timeseries
avgBOLD_fname = 'avgBOLD_cerebellum.txt';
EtCO2_t = TR * linspace(1, length(bestEtCO2), length(bestEtCO2))';
fileID_avgBOLD = fopen(avgBOLD_fname, 'w');
for zz = 1:length(EtCO2_t)
    fprintf(fileID_avgBOLD, '%f,%f\n', EtCO2_t(zz), bestEtCO2(zz));
end
fclose(fileID_avgBOLD);

%% Motion parameters and censor mask from fMRIPrep
rp_data = load([wkdir, filesep, subname, filesep, 'rp_fmriprep.txt']);
keep = logical(load([wkdir, filesep, subname, filesep, 'censor_fmriprep.txt']));
if size(rp_data, 1) ~= dynnum || numel(keep) ~= dynnum
    error('Motion (%d) or censor (%d) length does not match BOLD (%d)', size(rp_data,1), numel(keep), dynnum);
end
fprintf('Motion covariates: 6 parameters + squares; censored volumes: %d of %d\n\n', sum(~keep), dynnum);

%% Calculate RS BAT map using original function
fprintf('Running voxel-wise BAT GLM analysis...\n');
fprintf('This may take 10-30 minutes...\n\n');

cvrdir_v = [wkdir, filesep, subname, filesep, 'CVR_voxelshift_etco2_cerebellum'];
if ~exist(cvrdir_v, 'dir')
    mkdir(cvrdir_v);
end

maskImg = spm_read_vols(spm_vol([wkdir, filesep, subname, filesep, 'mask', filesep, 'brainMask_RS.nii']));
fixedDelayRange(1) = -9;
fixedDelayRange(2) = 9;

% Call the ORIGINAL CVR mapping function
voxelwiseResult = CVR_mapping_voxelwise_GLM_RS(...
    [wkdir, filesep, subname, filesep, bold_file_dir], ...
    ['dr' bold_file_name], TR, SmoothFWHMmm, fixedDelayRange, maskImg, ...
    avgBOLD_fname, EtCO2_mean, EtCO2_min, cvrdir_v, rp_data, keep);

fprintf('\nBAT calculation complete\n');
fprintf('Output saved to: %s\n\n', cvrdir_v);

close all;

end
