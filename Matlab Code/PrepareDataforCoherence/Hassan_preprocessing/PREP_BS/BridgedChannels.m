function [PARAM] = BridgedChannels(sFiles,PARAM)
%BRIDGEDCHANNELS - Detect bridged channels based on neighbouring channels high correlation
%
% SYNOPSIS: [resFiles,PARAM] = BridgedChannels(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:    SEE fieldtrip: ft_prepare_neighbours.m --> give neighbours /r distance
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
% Created on: 14-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('PREP>_____DETECTING BRIDGED CHANNELS\n')

tic
% Read channel file
sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
channelNames = {sChan.Channel.Name}';
% Keep EEG channels only
% eegChan = find(strcmpi({sChan.Channel.Type},'EEG'));

bridgeMat = zeros(size(channelNames,1)); % Counter for bridged channels
bridges = [];   % Bridged channels per file
% TimeStamps = string();
for iFile = 1:length(sFiles)
    if iFile == 1 || any(bridges(:))
        fprintf('PREP>    Detecting bridges in epoch (%d/%d)',iFile,length(sFiles))
    else
        PrintProgress(iFile,length(sFiles));
    end
    
    eegChan = find(strcmpi({sChan.Channel.Type},'EEG'));

    % Read epoch file
    sData = in_bst_data(sFiles(iFile).FileName);
    % Keep only channels that were not flagged bad already
    eegChan = eegChan(sData.ChannelFlag(eegChan) == 1);
%     eegChan(sData.ChannelFlag ~= 1) = 0;
%     eegIdx = find(eegChan);
    % Get EEG characteristics (data, location, label, sample frequency)
    eeg.data = sData.F(eegChan,:)';
    eeg.loc = {sChan.Channel(eegChan).Loc};
    eeg.label = {sChan.Channel(eegChan).Name};
    eeg.fs = round(1 / (sData.Time(2) - sData.Time(1)));
    % filter EEG
%     [sigf,~,~] = bst_bandpass_filtfilt(sig,eeg.fs,0.3,35,0,'fir');
    % Add filtered data to eeg structure in orientation [time,channel]
%     eeg.data = sig';
    
    % Bridged channels if correlation r > 0.99 with neighbouring channels
    %   function modified from clean_channel.m
    %   ref: https://github.com/sccn/clean_rawdata/blob/master/clean_channels_nolocs.m
    % bridges: array of channel index [bridged_channel_1_1, bridged_channel_1_2]
    bridges = DetectBridges(eeg, PARAM.Channel);
    if bridges == -1;   return; end
    
    if any(bridges)
        % Get real index and names of bridged channels
        switch size(bridges,2)
            case 1
                % Should not happen since modif. on Oct. 8, 2019
                bridgedIdx = eegChan(bridges)';
                bridgedChanName = channelNames(bridgedIdx);
            case 2
                bridgedIdx = [eegChan(bridges(:,1))',eegChan(bridges(:,2))'];
                bridgedChanName = [channelNames(bridgedIdx(:,1)), ...
                    channelNames(bridgedIdx(:,2))];
            otherwise
                disp('ERROR IN DetectBridges.m output size...')
        end
        % Print message to advise which channels are flagged as bridged
        msg = 'Highly correlated (bridged) channels: ';
        for iBridge = 1:size(bridgedIdx,1)
            bridgeMat(bridgedIdx(iBridge,1),bridgedIdx(iBridge,2)) = ...
                bridgeMat(bridgedIdx(iBridge,1),bridgedIdx(iBridge,2)) + 1;
            if iBridge == 1
                msg = [msg bridgedChanName{iBridge,1} '-' ...
                    bridgedChanName{iBridge,2}];
            else
                msg = [msg ', ' bridgedChanName{iBridge,1} '-' ...
                    bridgedChanName{iBridge,2}];
            end
        end
        fprintf('\nPREP>\t\t%d %s\n',size(bridgedIdx,1), msg);
    end
end

% Apply bridge duration threshold across epoch
nBridgedEpThresh = ceil(PARAM.Channel.min_corr_time * length(sFiles)); %min_corr_time =0.8
[r,c,v] = find(bridgeMat >= nBridgedEpThresh);

% Mark bridged channels as bad for all epoch files
if ~isempty(v)
    chan2Flag = unique([r,c]);
    chan2Flag = strjoin(channelNames(chan2Flag),',');
    % Set bad channel flag
    
    bst_process('CallProcess', 'process_channel_setbad', sFiles, [], ...
        'sensortypes', chan2Flag);
end

bst_report('Save', sFiles);
% TimeStamps = string();
% % info prep for excel
% if ~isempty(h)
% rep = load(h);
% x = (1:iFile)';
% TimeStamps((1:iFile),1)= "Bridged Channel";
% TimeStamps((1:iFile),2) = x;
% TimeStamps((1:iFile),3)= {sFiles.Comment};
% TimeStamps((1:iFile),4)= rep.Reports{1,2}.options.sensortypes.Value; 
% TimeStamps = cellstr(TimeStamps);
end
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | DetectBridging'],toc);
        