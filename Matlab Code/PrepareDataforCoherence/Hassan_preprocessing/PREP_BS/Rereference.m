function [resFiles,PARAM] = Rereference(sFiles,PARAM)
%REREFERENCE - Re-reference EEG recordings
%
% SYNOPSIS: [resFiles,PARAM] = Rereference(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 
%
% Copyright Tomy Aumont

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1135713 (R2019a) Update 3 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 23-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
disp('PREP>_____RE-REFERENCING...');

% Check if there is already a re-referencing projector
ChannelMat = in_bst_channel(sFiles.ChannelFile);
if ~isempty(ChannelMat.Projector) && any(cellfun(@(c)isequal(c,'REF'), {ChannelMat.Projector.SingVal}))
    disp('PREP> WARNING: There was already a re-referencing projector. Skipping');
    resFiles = sFiles;
    return
end
% Re-reference
resFiles = bst_process('CallProcess', 'process_eegref', sFiles, [], ...
        'eegref',      PARAM.Reref, ...
        'sensortypes', 'EEG');
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
	[PARAM.currentSubject ' | Rereference'],toc);    