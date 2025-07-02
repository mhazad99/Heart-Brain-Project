function [outFiles,PARAM] = GetRawFiles(PARAM)
%GETRAWFILES - Given a recordings corresponding to a subject list
%
% SYNOPSIS: [outFiles,PARAM] = GetRawFiles(PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1099231 (R2019a) Update 1 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 10-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
disp('PREP>_____SELECTING INPUT FILES')

% Select type of recordings
fileTypes = {'BrainVision','EDF/EDF+'};
answer = questdlg('Choose a file type', ' Recording type menu', ...
	fileTypes{:},'Cancel',... % button choices
    fileTypes{1}); % default button

% where to find the files
disp('Select recording directory')
rawDir = uigetdir(pwd,'Select recording directory');

% Get data and marker files
switch answer
    case 'BrainVision'
        files.data = dir([rawDir filesep '*.eeg']);     % find .eeg files
        files.mrk = dir([rawDir filesep '*.vmrk']);   % find corresponding marker files (.vmrk files)
        
        % Keep only files pairs with exact same names but different extensions
        % Get file names without extensions
        dataFileNames = cellfun(@(c) c(1:end-4),{files.data.name},'UniformOutput',false);
        mrkFileNames = cellfun(@(c) c(1:end-5),{files.mrk.name},'UniformOutput',false);
        % Exlude marker files that are name differently than data file (suffix-wise)
        goodFile = ismember(mrkFileNames,dataFileNames);
        files.mrk = files.mrk(goodFile);
        % Exlude data file that do not have corresponding marker files
        goodFile = ismember(dataFileNames,mrkFileNames);
        files.data = files.data(goodFile);
        
    case 'EDF/EDF+'
        
        files.data = dir([rawDir filesep '*.edf']);                       % find .edf files
       %files.data = dir([rawDir filesep '*.rec']);     % find .rec files
%		files.data = [files.data dir([rawDir filesep '*.REC'])];
        if isempty(files.data)
            fprintf(2,'PREP> No recording file found.\n')
            outFiles = [];
            return
        end
        files.mrk = dir([rawDir filesep '*.vmrk']);   % find .vmrk files
        if isempty(files.data)
            fprintf(2,'PREP> No marker file found.\n')
            outFiles = [];
            return
        end
        % Keep only files pairs with exact same names but different extensions
        % Get file names without extensions
        dataFileNames = cellfun(@(c) c(1:end-4),{files.data.name},'UniformOutput',false);
        mrkFileNames = cellfun(@(c) c(1:end-5),{files.mrk.name},'UniformOutput',false);
        % Exlude marker files that are name differently than data file (suffix-wise)
        goodFile = ismember(mrkFileNames,dataFileNames);
        files.mrk = files.mrk(goodFile);
        % Exlude data file that do not have corresponding marker files
        goodFile = ismember(dataFileNames,mrkFileNames);
        files.data = files.data(goodFile);
        
end

% Keep only files corresponding to the defined subjects
keep = cellfun(@(x) contains(x,PARAM.Subjects),{files.data.name});
files = structfun(@(x) x(keep),files,'UniformOutput',false);


% Sort files to fit subjectNames order
for iSubj = 1:length(PARAM.Subjects)
    % Sort data files
    subjPos = contains({files.data.name},PARAM.Subjects{iSubj});
    outFiles.data(iSubj) = files.data(subjPos);
    
%     subjPos = contains({files.vhdr.name},subjectNames{iSubj});
%     outFiles.vhdr(iSubj) = files.vhdr(subjPos);

    % Sort marker files the same way as data files
    subjPos = contains({files.mrk.name},PARAM.Subjects{iSubj});
    outFiles.mrk(iSubj) = files.mrk(subjPos);
end

% Make absolute path to these files
outFiles = structfun(@(x) fullfile({x.folder},{x.name}),outFiles,'UniformOutput',false);

% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration,'All subjects | GetRawFiles',toc);