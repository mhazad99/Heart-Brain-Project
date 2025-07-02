function [resFile, PARAM] = ReviewAsRaw(sFiles,PARAM)
% SYNOPSIS: [resFile, PARAM] = ReviewAsRaw(sFiles,PARAM,subjectName)
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
% Created on: 31-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Start a new report
bst_report('Start', sFiles);
% Process: Review raw file
disp('PREP>_____REVIEWING AS RAW FILE...')

resFile = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
    'subjectname',    PARAM.currentSubject, ...
    'datafile',       {sFiles.FileName,'BST-DATA'}, ...
    'channelreplace', 1, ...
    'channelalign',   1, ...
    'evtmode',        'value');

if isempty(resFile)
    fprintf(2, 'PREP> ERROR: Could not review raw file...\n')
    % Save and display report
    ReportFile = bst_report('Save', sFiles);
    bst_report('Open', ReportFile);
    return
else
    PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
        [PARAM.currentSubject ' | ReviewRawConcatFile'],toc);
end
