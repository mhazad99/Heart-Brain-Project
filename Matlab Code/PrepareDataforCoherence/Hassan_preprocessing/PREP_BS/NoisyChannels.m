function [sFiles,PARAM, NoisyChannel] = NoisyChannels(sFiles,PARAM)
%NOISYCHANNELS - Detect noisy channels based on channel standard deviation 
%   compared to standard deviation of the mean of the other channels
%
% SYNOPSIS: [sFiles,PARAM] = NoisyChannels(sFiles,PARAM)
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
% Created on: 15-Aug-2019
% Revised on: 13-Aug-2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('PREP>_____NOISY CHANNELS...\n');tic

tic
% Get EEG electrode name(s)
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
% eegChan = strcmpi({sChan.Channel.Type},'EEG')';
b = 0;
timeStamps = [];
binNoisy = [];
badMask = [];
for iFile = 1:length(sFiles)

    eegChan = strcmpi({sChan.Channel.Type},'EEG')';

    % Read DataFile
    sData = in_bst_data(sFiles(iFile).FileName);
    centered_sign = sData.F - mean(sData.F,2);

    % Keep only channels that were not flagged bad already
    eegChan(sData.ChannelFlag ~= 1) = 0;
    goodEEGIdx = find(eegChan);
    
    R_bad = zeros(size(goodEEGIdx));
    % Compute discrepancy ratio of all channels
    for iChan = 1 :length(goodEEGIdx)

 
        % Mask current channel
        cMask = eegChan;
        cMask(goodEEGIdx(iChan)) = 0;
        % Compute mean of good EEG across channels without the investiguated
%         M_others = mean(sData.F(cMask,:),1);
        M_others = mean(centered_sign(cMask,:),1);

        % Compute std across time of current channel
%         cStd = std(sData.F(goodEEGIdx(iChan),:));
         cStd = std(centered_sign(goodEEGIdx(iChan),:));

        % Compute discrepancy ratio of current channel vs the others
        R_bad(iChan) = cStd / std(M_others);
    end
    
    % Flag noisy channels based on their discrepancy ratio
    flagged = R_bad > PARAM.Channel.noisy_ratio;
    if any(flagged)
        badChan = goodEEGIdx(flagged);
        badChNames = strjoin({sChan.Channel(badChan).Name},',');
        
        fprintf('PREP> Epoch %d -> %d noisy channels detected:\n\t\t%s\n', ...
            iFile, length(badChan),badChNames)
        
        b = b+1;
        a = badChan;
        testing = zeros(1,128);
        testing(a) = 1;            

        good_label = get_sleep_label(sData);

        badMask{b,1} = sData.History(3,3);
        badMask{b,2} = good_label;
        badMask{b,3} = iFile;
        badMask{b,4} = badChNames;
        badMask{b,5} = a;
        badMask{b,6} = ones(length(a), length(sData.Time));
        

        % Apply flags in DataFile structure
        bst_process('CallProcess','process_channel_setbad',sFiles(iFile),[],...
            'sensortypes', badChNames);
    end
end
% Save report
bst_report('Save', sFiles);

if ~isempty(badMask)
    NoisyChannel = struct( 'time_segment',badMask(:,1),...
        'sleep_label',badMask(:,2), ...
        'flagged_epoch',badMask(:,3),'flagged_label', badMask(:,4), ...
        'flagged_index', badMask(:,5), 'binary_Mask', badMask(:,6));

end 
if isempty(badMask)
    NoisyChannel = struct();
end


% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | DetectNoisyChannels'],toc);
