function [resFiles,PARAM] = DetectMovements(sFiles,PARAM)
%DETECTMOVEMENTS - Detect movement based on standard deviation of EMG channels.
%
% SYNOPSIS: [resFiles,PARAM] = DetectMovements(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS: GROUP OF EPOCHS NEEDED ???
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

disp('PREP>_____DETECTING MOVEMENTS ON EMG')

toContinuous = 0;

% Get EMG electrode name(s)
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
chanNames = {sChan.Channel(strcmpi({sChan.Channel.Type},'EMG')).Name};
chanNames =  join(chanNames,',');


if ndims(sFiles) == 1
    %% SPLIT CONTINUOUS FILE TO EPOCH
    fprintf('PREP>\tFile is continuous, splitting file in %d seconds epoch...\n',PARAM.EpochLength)
    % Start a new report
    bst_report('Start',sFiles);
    % Process: Import MEG/EEG: Time
    resFiles = bst_process('CallProcess', 'process_import_data_time', sFiles, [], ...
        'subjectname', PARAM.currentSubject, ...
        'condition',   'MovementDetection', ...
        'timewindow',  [], ...
        'split',       PARAM.Import.length, ...
        'ignoreshort', 0, ...
        'usectfcomp',  0, ...
        'usessp',      0, ...
        'freq',        [], ...
        'baseline',    []);
    
    % Save and display report
    ReportFile = bst_report('Save', resFiles);
    
    if isempty(resFiles)
        fprintf(2,'PREP> ERROR: Could not epoch file... Abort\n')
        return
    else
        if strcmpi(sFiles.FileType,'raw')
            toRaw = 1;
        else
            toRaw = 0;
        end
        sFiles = resFiles;
        toContinuous = 1;
    end
    
end

%% DETECT EMG MOVEMENT
% Start a new report
bst_report('Start', sFiles);tic
resFiles = bst_process('CallProcess', 'process_evt_detect', sFiles, [], ...
    'eventname',    'bad_HighStd_EMG', ...
    'channelname', chanNames{:}, ...
    'timewindow',   [], ...
    'bandpass',     [1 35], ... [] to process raw EMG else [10 50] for filtered
    'threshold',    PARAM.Artifact.std_tresh_chan, ...
    'blanking',     PARAM.Artifact.std_min_time_interval, ...
    'isnoisecheck', 1, ...
    'isclassify',   0);

PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | MovementDetect'],toc);
        
% Save and display report
ReportFile = bst_report('Save', sFiles);

if isempty(resFiles)
    disp('PREP> WARNING: Could not detect movements. See brainstorm report.')
    bst_report('Open', ReportFile);
elseif toContinuous
        [resFiles,PARAM] = Epoch2Continuous(resFiles,PARAM,PARAM.currentSubject);
        if toRaw
            [resFiles, PARAM] = ReviewAsRaw(resFiles,PARAM,PARAM.currentSubject);
        end
end
