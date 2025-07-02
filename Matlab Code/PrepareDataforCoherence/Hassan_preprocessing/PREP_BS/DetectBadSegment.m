function [resFiles,PARAM] = DetectBadSegment(sFiles,PARAM)
%DETECTBADSEGMENT - Detect bad segment in the 1-7 Hz frequency band
%
% SYNOPSIS: [resFiles,PARAM] = DetectBadSegment(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:  1. Detect event on EEG to correct with SSP.
%                      2. Detect again and mark as bad
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

disp('PREP>_____DETECTING BAD SEGMENT [1-7] Hz')

tic
resFiles = bst_process('CallProcess', 'process_evt_detect_badsegment', sFiles, [], ...
            'timewindow',  [], ...
            'sensortypes', 'EEG', ...
            'threshold',   5, ...     1 is very sensitive, 5 is conservative
            'isLowFreq',   1, ...   detect 1-7 Hz artifact: eye movement/muscular
            'isHighFreq',  0);      % detect 40-240 Hz artifacts: electrodes & sensors artifacts
        
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | EEG_1-7Hz'],toc);
        