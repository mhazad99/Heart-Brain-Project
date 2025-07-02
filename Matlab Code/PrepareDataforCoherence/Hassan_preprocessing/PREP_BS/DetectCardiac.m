function [resFiles,PARAM] = DetectCardiac(sFiles,PARAM)
% SYNOPSIS: [resFiles,PARAM] = DetectCardiac(sFiles,PARAM)
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

disp('PREP>_____DETECTING CARDIAC EVENTS')

% Get cardiac electrode name(s)
sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
chanNames = {sChan.Channel(strcmpi({sChan.Channel.Type},'ECG')).Name};
chanNames =  join(chanNames,',');
if isempty(chanNames)
    fprintf(2,'PREP> ERROR: No ECG channel detected.\n')
    resFiles = [];
    return
end

% Start a new report
bst_report('Start', sFiles);tic
% Detect cardiac event
resFiles = bst_process('CallProcess', 'process_evt_detect_ecg', sFiles, [], ...
        'channelname', chanNames{:}, ...
        'timewindow',  [], ...
        'eventname',   'cardiac');
% Log processing time
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | DetectCardiac'],toc);
% Save and display report
ReportFile = bst_report('Save', sFiles);
% Display warning and brainstorm report if needed
if isempty(resFiles)
    disp('PREP> WARNING: Could not detect heartbeats. See brainstorm report.')
%     bst_report('Open', ReportFile);
end
