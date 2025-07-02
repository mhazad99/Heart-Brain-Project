function [resFiles,PARAM] = BadEEGSegment(sFiles,PARAM)
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

% % Process: Detect: BAD_EEG
% chanNames =  join(chanNames,',');
% resFiles = bst_process('CallProcess', 'process_evt_detect', sFiles, [], ...
%     'eventname',    'BAD_EEG', ...
%     'channelname',  chanNames{:}, ... 'Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fpz,Fz,Cz,CPz,Pz,POz,Oz,Iz,AF3,AF4,F1,F2,F5,F6,FC1,FC2,FC3,FC4,FC5,FC6,FT7,FT8,C1,C2,C5,C6,CP1,CP2,CP3,CP4,CP5,CP6,TP7,TP8,P1,P2,P5,P6,PO3,PO4,AFp1,AFp2,AF7,AF8,AFF1h,AFF2h,AFF5h,AFF6h,FFC1h,FFC2h,FFC3h,FFC4h,FFC5h,FFC6h,FFT7h,FFT8h,FFT9h,FFT10h,FT9,FT10,FCC1h,FCC2h,FCC3h,FCC4h,FCC5h,FCC6h,FTT7h,FTT8h,CCP1h,CCP2h,CCP3h,CCP4h,CCP5h,CCP6h,TTP7h,TTP8h,CPP1h,CPP2h,CPP3h,CPP4h,CPP5h,CPP6h,TPP7h,TPP8h,P9,P10,PPO1h,PPO2h,PPO5h,PPO6h,PPO9h,PPO10h,PO7,PO8,PO9,PO10,POO1,POO2,POO9h,POO10h,OI1h,OI2h,O9,O10', ...
%     'timewindow',   [], ...
%     'bandpass',     [0.5 35], ...
%     'threshold',    PARAM.Artifact.EEG_std_thresh, ...
%     'blanking',     PARAM.Artifact.EEG_min_time_interval, ... in seconds
%     'isnoisecheck', 0, ...
%     'isclassify',   0);

% Process ONE epoch at a time
for iFile = 1:length(sFiles)
    fprintf('PREP>\tDetecting bad EEG segment in epoch (%d/%d)\n',iFile,length(sFiles))
	
    sData = in_bst_data(sFiles(iFile).FileName);
    goodChan = eegIdx(sData.ChannelFlag(eegIdx) == 1);
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
    % Process ONE channel at a time
    for iChan = 1:length(goodChan)
        if iChan == 1
            fprintf('PREP>\t\tchannel (%d/%d)',iChan,length(goodChan));
        else
            PrintProgress(iChan,length(goodChan),true); % always new line because of Brainstorm display
		end
        % Process: Detect: highAmp per channels
        resFiles(iFile) = bst_process('CallProcess', 'process_ch_evt_detect_threshold', sFiles(iFile), [], ...
            'eventname',    'bad_HighAmp', ... %BAD_highAmp
            'channelname',  chanNames{goodChan(iChan)}, ...
            'timewindow',   timeWindow, ...
            'thresholdMAX', maxThresh(iChan), ...
            'units',        1, ...  % already in uV: 10e-6
            'bandpass',     [], ...
            'isAbsolute',   1, ...
            'isDCremove',   1);
        
%         resFiles(iFile) = bst_process('CallProcess', 'process_evt_detect', sFiles(iFile), [], ...
%             'eventname',    'BAD_EEG', ...
%             'channelname',  chanNames{goodChan(iChan)}, ... 'Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fpz,Fz,Cz,CPz,Pz,POz,Oz,Iz,AF3,AF4,F1,F2,F5,F6,FC1,FC2,FC3,FC4,FC5,FC6,FT7,FT8,C1,C2,C5,C6,CP1,CP2,CP3,CP4,CP5,CP6,TP7,TP8,P1,P2,P5,P6,PO3,PO4,AFp1,AFp2,AF7,AF8,AFF1h,AFF2h,AFF5h,AFF6h,FFC1h,FFC2h,FFC3h,FFC4h,FFC5h,FFC6h,FFT7h,FFT8h,FFT9h,FFT10h,FT9,FT10,FCC1h,FCC2h,FCC3h,FCC4h,FCC5h,FCC6h,FTT7h,FTT8h,CCP1h,CCP2h,CCP3h,CCP4h,CCP5h,CCP6h,TTP7h,TTP8h,CPP1h,CPP2h,CPP3h,CPP4h,CPP5h,CPP6h,TPP7h,TPP8h,P9,P10,PPO1h,PPO2h,PPO5h,PPO6h,PPO9h,PPO10h,PO7,PO8,PO9,PO10,POO1,POO2,POO9h,POO10h,OI1h,OI2h,O9,O10', ...
%             'timewindow',   timeWindow, ...
%             'bandpass',     [], ...
%             'threshold',    PARAM.Artifact.EEG_std_thresh, ...
%             'blanking',     PARAM.Artifact.EEG_min_time_interval, ... in seconds
%             'isnoisecheck', 0, ...
%             'isclassify',   0);
    end
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
