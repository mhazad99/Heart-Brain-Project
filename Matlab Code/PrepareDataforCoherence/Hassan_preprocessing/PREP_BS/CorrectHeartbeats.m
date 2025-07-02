function [resFiles,PARAM] = CorrectHeartbeats(sFiles,PARAM)
%CORRECTHEARTBEATS - Remove heartbeat artifact using SSP (PCA) projector on EEG
%
% SYNOPSIS: [resFiles,PARAM] = CorrectHeartbeats(sFiles,PARAM);
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
% Created on: 01-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('PREP>_____REMOVING HEARTBEATS')

% Start a new report
bst_report('Start',sFiles);

% Process: SSP ECG: cardiac => SSP2 for files not long enough
resFiles = bst_process('CallProcess', 'process_ssp_ecg', sFiles,  [], ...
    'eventname',   'cardiac', ...
    'sensortypes', 'EEG', ...
    'usessp',      1, ...
    'select',      1);

% Save and display report
ReportFile = bst_report('Save', resFiles);

PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | RemoveHeartbeat'],toc);

if isempty(resFiles)
    fprintf(2,'PREP> ERROR: Could not remove heartbeats\n')
end

