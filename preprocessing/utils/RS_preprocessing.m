function RS_preprocessing(wkdir, folder_name, code_directory)


cd([wkdir, filesep, folder_name]);
para_file_ID = fopen('parameter_RS.txt', 'r');


while ~feof(para_file_ID)

    tline = fgetl(para_file_ID);
    if regexp(tline, 'SmoothFWHM')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, ':');
        SmoothFWHMmm = str2num(tline(colon_loc+1:end));
    end

    if regexp(tline, 'TR')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, ':');
        TR = str2num(tline(colon_loc+1:end));
    end

    if regexp(tline, 'mprageFile')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        [mpr_file_dir, mpr_file_name, mpr_file_ext] = fileparts(tline(colon_loc(3)+1:colon_loc(4)-1));
    end

    if regexp(tline, 'boldFileName_RS')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        [bold_file_dir, bold_file_name, bold_file_ext] = fileparts(tline(colon_loc(3)+1:colon_loc(4)-1));
    end

    if regexp(tline, 'sliceOrderFile_RS')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        slice_order_file = tline(colon_loc(3)+1:colon_loc(4)-1);
    end

    if regexp(tline, 'refSlice')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, ':');
        ref_slice = str2num(tline(colon_loc+1:end));
    end
end


%% Global Parameter Setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath([code_directory, filesep, 'utils']);
addpath([code_directory, filesep, 'spm12']);

spm_get_defaults;
global defaults;
defaults.mask.thresh = 0;
mni_resolution = [2, 2, 2];
mni_type = 16;
envelope_interp_rate = 10;
tpm_loc = [code_directory, filesep, 'spm12', filesep, 'tpm']; %brain normlization template location

%% Detect file format (NIfTI .nii or ANALYZE .img) %%%%%%%%%%%%%%%%%%%%%%%%
rs_dir = [wkdir, filesep, folder_name, filesep, bold_file_dir];
cd(rs_dir);

% Auto-detect file format
if exist([rs_dir, filesep, bold_file_name, '.nii'], 'file')
    file_ext = 'nii';
    boldfiles = dir([rs_dir, filesep, bold_file_name, '.nii']);
    disp('Detected NIfTI format (.nii)');
elseif exist([rs_dir, filesep, bold_file_name, '.img'], 'file')
    file_ext = 'img';
    boldfiles = dir([rs_dir, filesep, bold_file_name, '*.img']);
    disp('Detected ANALYZE format (.img)');
else
    error('BOLD file not found: %s in %s', bold_file_name, rs_dir);
end


%% Realign BOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get original bold scan
P      = cell(1,1);
P{1}   = spm_select('FPList', rs_dir, file_ext, ['^', bold_file_name]);

if isempty(P{1})
    error('No BOLD files found with extension .%s and name %s', file_ext, bold_file_name);
end

% get the bold scan's data
V       = spm_vol(P);
V       = cat(1,V{:});

% realign bold scan
disp(sprintf(['realigning ' bold_file_name]));
FlagsC = struct('quality',defaults.realign.estimate.quality,...
    'fwhm',5,'rtm',0);
spm_realign(V, FlagsC);

% reslice bold scan
which_writerealign = 2;
mean_writerealign = 1;
FlagsR = struct('interp',defaults.realign.write.interp,...
    'wrap',defaults.realign.write.wrap,...
    'mask',defaults.realign.write.mask,...
    'which',which_writerealign,'mean',mean_writerealign);
spm_reslice(P,FlagsR);

%% Slice Timing Correction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% skip this step if sliceOrderFile is omitted
sliceOrderFile = [wkdir, filesep, folder_name, filesep, slice_order_file];
P = spm_select('FPList', rs_dir, file_ext, ['^r', bold_file_name]);

if ~isempty(sliceOrderFile) && ~isempty(P)

    % get original bold scan
    % set TA and refSlice
    scanInfo = spm_vol(P(1,:));  %use first dynamic to get number of slices
    nslices = scanInfo(1).dim(3);
    TA = TR-(TR/nslices);

    % get order of slices
    fid = fopen([sliceOrderFile]);
    sliceOrder = textscan(fid,'%f');
    sliceOrder = sliceOrder{1};
    fclose(fid);

    % get timing for correction
    timing(2)=TR-TA;
    timing(1)=TA/(nslices-1);

    % correct slice timing (produces a new bold scan with the filename prefixed
    % with 'a')
    spm_slice_timing(P, sliceOrder', ref_slice, timing);
end

% Find the processed BOLD file (after realignment and slice timing)
if strcmp(file_ext, 'nii')
    ar_bold_file = [rs_dir, filesep, 'ar', bold_file_name, '.nii'];
else
    ar_bold_file = [rs_dir, filesep, 'ar', boldfiles.name];
end

if ~exist(ar_bold_file, 'file')
    error('Processed BOLD file not found: %s', ar_bold_file);
end

arbold_image_header = spm_vol(ar_bold_file);
arbold_image = spm_read_vols(arbold_image_header);

for ii = 1:size(arbold_image, 4)
    if ii < 10
        write_hdrimg(arbold_image(:, :, :, ii), [rs_dir, filesep, 'ar', bold_file_name, '-00', num2str(ii), '-001.img'], mri_resolution(arbold_image_header(ii).mat), mni_type);
    elseif ii < 100
        write_hdrimg(arbold_image(:, :, :, ii), [rs_dir, filesep, 'ar', bold_file_name, '-0', num2str(ii), '-001.img'], mri_resolution(arbold_image_header(ii).mat), mni_type);
    else
        write_hdrimg(arbold_image(:, :, :, ii), [rs_dir, filesep, 'ar', bold_file_name, '-', num2str(ii), '-001.img'], mri_resolution(arbold_image_header(ii).mat), mni_type);
    end
end

%% MPRAGE File %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mpr_dir = [wkdir, filesep, folder_name, filesep, mpr_file_dir];
mpr_type = 16;
copyfile(mpr_dir, [mpr_dir, '_RS']);
mpr_dir = [mpr_dir, '_RS'];
cd(mpr_dir)

% Auto-detect MPRAGE format
if exist([mpr_file_name, '.nii'], 'file')
    mpr_ext = '.nii';
elseif exist([mpr_file_name, '.img'], 'file')
    mpr_ext = '.img';
else
    % Check for .nii.gz
    if exist([mpr_file_name, '.nii.gz'], 'file')
        disp('Decompressing .nii.gz file...');
        gunzip([mpr_file_name, '.nii.gz']);
        mpr_ext = '.nii';
    else
        error('MPRAGE file not found: %s', mpr_file_name);
    end
end

mpr_brain_vol = spm_read_vols(spm_vol([mpr_file_name, mpr_ext]));
[outVol,varargout] = reorientVol(mpr_brain_vol, '+x+y+z');
mpr_resolution = mri_resolution(spm_vol([mpr_file_name, mpr_ext]).mat);
write_hdrimg(outVol, [mpr_file_name, '.img'], mpr_resolution, mpr_type);
write_hdrimg(outVol, [mpr_file_name, '.nii'], mpr_resolution, mpr_type);

%% Coregistration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Segment MPRAGE and Create preMask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(mpr_dir)
segment_job([mpr_file_name, '.nii'], tpm_loc); %segment mprage image
for ii = 1:3
    if ii == 1
        mask_p = spm_read_vols(spm_vol([mpr_dir, filesep, 'c', num2str(ii), mpr_file_name, '.nii']));
    else
        mask_p = mask_p + spm_read_vols(spm_vol([mpr_dir, filesep, 'c', num2str(ii), mpr_file_name, '.nii']));
    end
end

new_info = spm_vol([mpr_file_name, '.nii']);
mpr_vox = mri_resolution(new_info.mat);
mask_p(mask_p>0.05) = 1;
write_hdrimg(mask_p, 'mask_p.nii', mpr_vox, mni_type);
delete(['y_', mpr_file_name, '.nii']);
delete([mpr_file_name, '_seg8.mat']);

mp_data = spm_read_vols(spm_vol([mpr_dir, filesep, mpr_file_name, '.nii']));
mp_data = mp_data.*mask_p;
mpr_mask_name = [mpr_dir, filesep, mpr_file_name, '_mask.nii'];
write_hdrimg(mp_data, mpr_mask_name, mpr_vox, mni_type);

%coregister MPRAGE to BOLD image
coreg_target = spm_select('FPList', rs_dir, ['^mean', bold_file_name]);
coreg_other = {[mpr_dir, filesep, mpr_file_name, '.nii']};
coreg_job(coreg_target, mpr_mask_name, coreg_other);

%% Segment MPRAGE and Create Mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(mpr_dir)
segment_job([mpr_file_name, '.nii'], tpm_loc); %segment mprage image

%% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(file_ext, 'nii')
    % For NIfTI, get number of volumes
    V_tmp = spm_vol([rs_dir, filesep, 'ar', bold_file_name, '.nii']);
    rs_dynnum = length(V_tmp);
else
    rs_dynnum = length(spm_vol([boldfiles.folder, filesep, 'ar', boldfiles.name]));
end

deform_field = spm_select('FPList', mpr_dir, ['^y_', mpr_file_name, '.nii']);
norm_job(rs_dir, bold_file_name, rs_dynnum, tpm_loc); %resoultion = 2*2*2mm

anaFile = spm_select('FPList', mpr_dir, ['^m', mpr_file_name, '.nii']);
norm_ana_job(anaFile, deform_field); %resoultion = 2*2*2mm

%% Skull-stripped Brain Mask%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
suit_isolate_seg({['wm', mpr_file_name, '.nii']}); %segement cerebellum
segment_job(['wm', mpr_file_name, '.nii'], tpm_loc); %segment mprage image

% segmented probability map
norm_segmentedList = dir(['c*wm' mpr_file_name, '.nii']);

for ii = 1:3 %GM, WM and CSF
    if ii == 1
        brainTissue = spm_read_vols(spm_vol(norm_segmentedList(ii).name));
    else
        brainTissue = brainTissue + spm_read_vols(spm_vol(norm_segmentedList(ii).name));
    end
end

levelBW = 0.8; %set up the threshold for brain tissue mask
brainMask = brainTissue > levelBW;

% write final brain mask
mpr_vox = mri_resolution(spm_vol(['wm', mpr_file_name, '.nii']).mat);
brainMaskfullName = [wkdir, filesep, folder_name, filesep, 'mask'];
mkdir(brainMaskfullName);
brainMaskfullName = [brainMaskfullName, filesep, 'brainMask_RS.nii'];
write_hdrimg(brainMask, brainMaskfullName, mpr_vox, mni_type);

% cerebellum mask
c_mask = spm_read_vols(spm_vol(['c1', 'wm', mpr_file_name, '.nii']));
c_mask = c_mask + spm_read_vols(spm_vol(['c2', 'wm', mpr_file_name, '.nii']));
c_mask(c_mask>0.95) = 1;
c_mask(c_mask<=0.95) = 0;
c_maskfullname = [wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_RS_cerebellum.nii'];
write_hdrimg(c_mask, c_maskfullname, mpr_vox, mni_type);

end
