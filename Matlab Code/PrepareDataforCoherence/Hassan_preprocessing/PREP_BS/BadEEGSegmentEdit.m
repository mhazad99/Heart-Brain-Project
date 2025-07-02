function [resFiles,PARAM] = BadEEGSegmentEdit(sFiles,PARAM)
%DETECTMOVEMENTS - Detect movement based on standard deviation of EMG channels.
%
% SYNOPSIS: [resFiles,PARAM] = DetectMovements(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS: GROUP OF EPOCHS NEEDED
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
% Created on: 05-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
% Start a new report
bst_report('Start', sFiles);

% Get EEG electrode name(s)
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
chanNames = {sChan.Channel.Name};

% Keep only EEG channels (without reference if re-referenced)
refNames = split(PARAM.Reref,',')';
eegIdx = find(strcmpi({sChan.Channel.Type},'EEG') & ...
    ~cellfun(@(c) any(find(strcmpi(refNames,c))), {sChan.Channel.Name}));

goodChanNames = {sChan.Channel.Name};
% goodChanNames = cellstr(goodChanNames);   %nope
goodChanNames2 = char(join(goodChanNames,','));
goodChanNames3 = split(string(goodChanNames2),',');    %nope
% Process ONE epoch at a time
for iFile = 1:length(sFiles)
    
    fprintf('PREP>\tDetecting bad EEG segment in epoch (%d/%d)\n',iFile,length(sFiles))
    
    sData = in_bst_data(sFiles(iFile).FileName);
%     goodChan = eegIdx(sData.ChannelFlag(eegIdx) == 1);
     goodChan = [1:128];


    %     goodChan = sData.ChannelFlag(eegIdx) == 1;
    %     goodIdx = find(goodChan);
    % remove first and last second from processing
    timeWindow = [sData.Time(1)+1, sData.Time(end)-1];
    fs = round(1 / (sData.Time(2) - sData.Time(1))); % same quantity in samples
    
    chanMean = mean(sData.F(goodChan,fs:end-fs),2);
    chanStd = std(sData.F(goodChan,fs:end-fs),[],2);
    % thresh = n * std + mean
    maxThresh = PARAM.Artifact.EEG_std_thresh * chanStd + chanMean;
    %     maxThresh = maxThresh(eegIdx);
    % Process: Detect: BAD_EEG
    
    resFiles(iFile) = bst_process('CallProcess', 'process_ch_evt_detect_threshold', sFiles(iFile), [], ...
        'eventname',    'bad_HighAmp', ... %BAD_highAmp
        'channelname',  goodChanNames2, ...
        'timewindow',   timeWindow, ...
        'thresholdMAX', maxThresh, ...
        'units',        1, ...  % already in uV: 10e-6
        'bandpass',     [], ...
        'isAbsolute',   0, ...
        'isDCremove',   0);    
end


toc
% Save and display report
ReportFile = bst_report('Save', sFiles);
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | DetectBadEEGsegment'],toc);
% Open report if problem happened
if isempty(resFiles)
    disp('PREP> WARNING: Could not detect bad EEG segment. See brainstorm report.')
    bst_report('Open', ReportFile);
end
