function  [resFiles,PARAM] = CorrectEyeMovement(sFiles,PARAM)
%CORRECTEYEMOVEMENT - Rename "1-7Hz" events to EyeMovement and remove them using SSP (PCA)
%
% SYNOPSIS:  [resFiles,PARAM] = CorrectEyeMovement(sFiles,PARAM)
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

disp('PREP>_____REMOVING EYE MOVEMENT')

% Start a new report
bst_report('Start',sFiles);

% Process: Rename event
fprintf('PREP>\tRenaming events "1-7Hz" to EyeMovement\n')
resFiles = bst_process('CallProcess', 'process_evt_rename', sFiles, [], ...
    'src',  '1-7Hz', ...
    'dest', 'EyeMovement');

% Process: SSP EyeMovement: 1-7Hz
fprintf('PREP>\tComputing PCA for EyeMovement')
resFiles = bst_process('CallProcess', 'process_ssp', resFiles,[], ...
    'timewindow',  [], ...
    'eventname',   'EyeMovement', ...
    'eventtime',   [-0.2, 0.2], ... 200 ms before and after peak
    'bandpass',    [1, 7], ...      event detection frequency band
    'sensortypes', 'EEG', ...
    'usessp',      0, ...
    'saveerp',     0, ...
    'method',      1, ...               PCA: One component per sensor
    'select',      1 ...                  select first component as active
    );

% Save and display report
ReportFile = bst_report('Save', resFiles);

PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | RemoveEyeMovement'],toc);

if isempty(resFiles)
    fprintf(2,'PREP> ERROR: Could not remove eye movements\n')
end
        
        