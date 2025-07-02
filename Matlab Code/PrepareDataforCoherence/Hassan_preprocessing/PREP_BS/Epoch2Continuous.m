function [resFile, PARAM] = Epoch2Continuous(sFiles,PARAM)
%EPOCH2CONTINUOUS - Concatenate epoch into a continuous file. First epoch give the time reference.
%
% SYNOPSIS: [resFile, PARAM] = Epoch2Continuous(sFiles,PARAM,subjectName)
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
% Created on: 29-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start a new report
bst_report('Start', sFiles);

% Process: Concatenate time
disp('PREP>_____CONCATENATING EPOCH...')
fprintf('\t\tThis step may be long. Please be patient...\n')
tic
resFile = bst_process('CallProcess', 'process_concat', sFiles, []);
if isempty(resFile)
    fprintf(2, 'PREP> ERROR: Could not concatenate epoch into continous file\n')
    % Save and display report
    ReportFile = bst_report('Save', sFiles);
    bst_report('Open', ReportFile);
    return
else
    PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
        [PARAM.currentSubject ' | ConcatenateEpoch'],toc);
end




