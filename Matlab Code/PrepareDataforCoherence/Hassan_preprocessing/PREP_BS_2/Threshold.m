function [sFiles,PARAM,ThresholdChannel] = Threshold(sFiles,PARAM)
% Threshold - Detect peak amplitude per channel above threshold for more than 10ms
%             Thresold : n*cStd + cMean
%
% SYNOPSIS: [sFiles,PARAM] = Threshold(sFiles,PARAM)
%
%
fprintf('PREP>_____THRESHOLD...\n');tic

% Get EEG electrode name(s)
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
% eegChan = strcmpi({sChan.Channel.Type},'EEG')';
b=0;
timestamps = [];
binThresh = [];
for iFile = 1:length(sFiles)

    eegChan = strcmpi({sChan.Channel.Type},'EEG')';
    
    sData = in_bst_data(sFiles(iFile).FileName);
    eegChan(sData.ChannelFlag ~= 1) = 0;
    
    %index of all good channel
    goodEEGIdx = find(eegChan);
    
    %sample frequency
    fs = round(1/ (sData.Time(2) - sData.Time(1)));  %test with round
    
    %compute DC offset
    sign = computeDCoffset(sData.F,eegChan);
    
    %compute n*std +mean
    cMean = mean(sign(:,:),2);
    cStd = std(sign(:,:),[],2);
    maxEEGThresh = PARAM.Artifact.EEG_std_thresh*cStd + cMean;
    
    
    sign = abs(sign);
    % signal above threshold
    c = zeros(size(sign));
    c(sign > maxEEGThresh) = 1;
    d = sum(c,2);
    
    %10 ms above threshold
    t = 0.01;
    tat = t*fs;
    
    % Find channel over 10ms threshold
    flagged = ...
        d > tat; 
    
    %number of bad channel per epoch 
    % for information purposes only
    nbfOfBadElectrodes = sum(flagged,1);
    
    %flag bad channels (brainstorm)
    if any(flagged)
        %         badChan = goodEEGIdx(flagged);
        
        chNames = strjoin({sChan.Channel(goodEEGIdx(flagged)).Name},',');
        
        b = b+1;
      
        maskthresh = c(flagged,:);
        a = goodEEGIdx(flagged);
        testing = zeros(1,128);
        testing(a) = 1;

        good_label = get_sleep_label(sData);

        
        badMask{b,1} = sData.History(3,3);
        badMask{b,2} = good_label;  
        badMask{b,3} = iFile;
        badMask{b,4} = chNames;
        badMask{b,5} = a;
        badMask{b,6} = maskthresh;
        
%         fprintf('PREP> Epoch %d -> %d bad segments detected:\n\t\t%s\n', ...
%             iFile, length(chNames),chNames)
        bst_process('CallProcess','process_channel_setbad',sFiles(iFile),[],...
            'sensortypes', chNames);
    end
end
if ~isempty(badMask)
    ThresholdChannel = struct( 'time_segment',badMask(:,1),...
        'sleep_label',badMask(:,2), ...
        'flagged_epoch',badMask(:,3),'flagged_label', badMask(:,4), ...
        'flagged_index', badMask(:,5), 'binary_Mask', badMask(:,6));
end


% Save report
bst_report('Save', sFiles);
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | threshold'],toc);