function [sFiles,PARAM,FlatChannel] = FlatChannel(sFiles,PARAM)
%------------------------------------------
% FlatChannel - Flat channel detection for low amplitude signals.
%               Based on two criterias:
%               1)Standard deviation > trf2
%               nbr of samples(T) where absolute of current channel  < 0.1*Standard deviation
%               2)T*Sample Period > trf2
%
% SYNOPSIS: [sFiles,PARAM] = FlatChannel(sFiles,PARAM)
%
%
%------------------------------------------
fprintf('PREP>_____FLATCHANNEL...\n');tic

% Get EEG electrode name(s)
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
% eegChan = strcmpi({sChan.Channel.Type},'EEG')';
b=0;
timeStamps = [];
binFlat = [];
badMask = [];
for iFile = 1:length(sFiles)
    
    eegChan = strcmpi({sChan.Channel.Type},'EEG')';
    
    sData = in_bst_data(sFiles(iFile).FileName);

%     for i = 1: size(sData.F,1)  
%         smoothed(i,:) = smooth(sData.F(i,:)',5);
%     end

   centered_sign = sData.F - mean(sData.F,2);
%     centered_sign = sData.F - smoothed;

    eegChan(sData.ChannelFlag ~= 1) = 0;
    
    %index of all good channel
    goodEEGIdx = find(eegChan);
    
    %compute Sample frequency (500)
    fs = round(1/ (sData.Time(2) - sData.Time(1)));  %test with round
    trf2 = 10e-6 ; % 10 microvolts
    trtf = 3;    % 0.8 seconds
    
    %compute std per channel
%     signstd = std(sData.F(eegChan,:),[],2);
    signstd = std(centered_sign(eegChan,:),[],2);
    %compute trAmp: std of signal * 0.1
    signTrAmp = signstd.*0.1;
    %compute abosulte of signal
%    signabs = abs(sData.F(eegChan,:));
    signabs = abs(centered_sign(eegChan,:));

    
    %Find samples where absolute of signal < signTrAmp
    % and set them as one
%     c = zeros(size(sData.F(eegChan,:)));
    c = zeros(size(centered_sign(eegChan,:)));
    c(signabs < signTrAmp) = 1;
    %  Get number of samples respecting the criteria for every channel
    numT = sum(c,2);
    %compute: nbr sample * sample period
    T = numT*(1/fs);
    
    % flat channel crtiteria
    flatChannel = ...
        (signstd < trf2  & T > trtf);
    
    %number of flat channel per epoch
    %for information purposes
    nbrFC = sum(flatChannel,1);
    
    %flag bad channels
    if any(flatChannel)
        chNames = strjoin({sChan.Channel(goodEEGIdx(flatChannel)).Name},',');
        
        b=b+1;
        a = goodEEGIdx(flatChannel);
        testing = zeros(1,128);
        
        testing(a) = 1;
        good_label = get_sleep_label(sData);

        maskFlat = c(flatChannel,:);
        badMask{b,1} = sData.History(3,3);
        badMask{b,2} = good_label;    
        badMask{b,3} = iFile;
        badMask{b,4} = chNames;
        badMask{b,5} = a;
        badMask{b,6} = maskFlat;
        
        bst_process('CallProcess','process_channel_setbad',sFiles(iFile),[],...
            'sensortypes', chNames);
    end
end
if ~isempty(badMask)
    FlatChannel = struct( 'time_segment',badMask(:,1),...
        'sleep_label',badMask(:,2), ...
        'flagged_epoch',badMask(:,3),'flagged_label', badMask(:,4), ...
        'flagged_index', badMask(:,5), 'binary_Mask', badMask(:,6));
end
if isempty(badMask)
    FlatChannel = struct();
end
% Save report
bst_report('Save', sFiles);
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | FlatChannel'],toc);
