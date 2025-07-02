function outFiles = GetRawFiles(subjectNames)
%GETRAWFILES - Given a recording directory, select BrainVision files 
%   corresponding to a subject
%
% SYNOPSIS: rawFiles = GetRawFiles(rawDir,subjectNames,eeg,vhdr,vmrk)
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

fileTypes = {'BrainVision (.eeg, .vmrk, .vhrd)','EDF/EDF+'};
[choiceIdx,tf] = listdlg('PromptString','Choose recording files type:', ...
    'SelectionMode','single', ...
    'ListString',fileTypes);
if ~tf
    fprintf('User cancelled sleep stage selection. Exit\n')
    return
end
fType = fileTypes(choiceIdx);

rawDir = uigetdir(pwd,'Select recording directory'); % where to find the files

files.data = dir([rawDir filesep '*.eeg']);     % find .eeg files
files.vhdr = dir([rawDir filesep '*.vhdr']);    % find .vhdr files
files.mrk = dir([rawDir filesep '*.vmrk']);   % find .vmrk files

% Do not select original or backup files if another one is present
uniq = ~(contains({files.mrk.name},'_orig') | contains({files.mrk.name},'_bckup'));
files.mrk = files.mrk(uniq);

% Keep only files corresponding to the defined subjects
keep = cellfun(@(x) contains(x,subjectNames),{files.data.name});
files = structfun(@(x) x(keep),files,'UniformOutput',false);

% Sort files to fit subjectNames order
for iSubj = 1:length(subjectNames)
    subjPos = contains({files.data.name},subjectNames{iSubj});
    outFiles.data(iSubj) = files.data(subjPos);
    
    subjPos = contains({files.vhdr.name},subjectNames{iSubj});
    outFiles.vhdr(iSubj) = files.vhdr(subjPos);
    
    subjPos = contains({files.mrk.name},subjectNames{iSubj});
    outFiles.mrk(iSubj) = files.mrk(subjPos);
end

% Make absolute path to these files
outFiles = structfun(@(x) fullfile({x.folder},{x.name}),outFiles,'UniformOutput',false);
