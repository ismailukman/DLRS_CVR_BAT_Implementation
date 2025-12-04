clc;
clear all;
close all;


%% Navigate to code folder and data folder
[code_directory, ~, ~] = fileparts(mfilename('fullpath'));
addpath(code_directory);
addpath([code_directory, filesep, 'utils']);

[parent_directory, ~, ~] = fileparts(code_directory);
data_directory = [parent_directory, filesep, 'data'];

subfolder = dir(data_directory);
% Remove hidden files, system files, and non-directories
subfolder(ismember({subfolder.name}, {'.', '..', '.gitkeeper', 'output', '.DS_Store'})) = [];
% Keep only directories
subfolder = subfolder([subfolder.isdir]);

% Only process AFNI preprocessed subjects (sub-03a, sub-07a)
valid_subjects = {};
for jj = 1:length(subfolder)
    if startsWith(subfolder(jj).name, 'sub-')
        valid_subjects{end+1} = subfolder(jj).name;
    end
end

%% Preprocessing pipeline
for ii=1:length(valid_subjects)
    fprintf('Processing subject: %s\n', valid_subjects{ii});
    RS_preprocessing(data_directory, valid_subjects{ii}, code_directory);
    RS_BAT(data_directory, valid_subjects{ii}, code_directory);
    RS_CVR_corrMap(data_directory, valid_subjects{ii}, code_directory);
end
