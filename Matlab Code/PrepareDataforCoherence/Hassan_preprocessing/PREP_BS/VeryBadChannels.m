function [resFiles,PARAM,ObviouslyBadChannel] = VeryBadChannels(sFiles,PARAM)
%VERYBADCHANNELS - Flag bad (noisy, flat) channels based on standard deviation.
%   If a channel's standard deviation per epoch is > Trn or < Trf, reject it for
%   the corresponding epoch.
% 
% SYNOPSIS: [resFiles,PARAM] = VeryBadChannels(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%   Reference: Dorothee Coppieters ’t Wallant, Vincenzo Muto, Giulia Gaggioni,
%       Mathieu Jaspar, Sarah L. Chellappa, Christelle Meyer, Gilles Vandewalle,
%       Pierre Maquet, Christophe Phillips, Automatic artifacts and arousals
%       detection in whole-night sleep EEG recordings, Journal of Neuroscience
%       Methods (2015), http://dx.doi.org/10.1016/j.jneumeth.2015.11.005
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
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('PREP>_____DETECTING VERY BAD CHANNELS...\n');tic

% Ensure not to return an error for not assigning output variable
resFiles = sFiles;

tic
% Get EEG electrode index
if ndims(sFiles) == 1
    sChan = in_bst_channel(sFiles.ChannelFile,'Channel');
else
    sChan = in_bst_channel(sFiles(1).ChannelFile,'Channel');
end
eegChan = strcmpi({sChan.Channel.Type},'EEG');
b = 0;
timeStamps = [];
binVeryBad = [];
badMask = [];
% flagged = zeros(length(find(eegChan)),length(sFiles));
for iFile = 1:length(sFiles)
    
    % Read DataFile
    sData = in_bst_data(sFiles(iFile).FileName);
    % Keep only EEG channels that were not flagged bad already
    eegChan(sData.ChannelFlag ~= 1) = 0;
    chanIdx = find(eegChan);
    
    % Compute standard deviation per channel
    centered_sign = sData.F(eegChan,:) - mean(sData.F(eegChan,:),2);
    chanStd = std(centered_sign,[],2);
    
    
    % Flag really bad channels
    flagBad = ...
        chanStd > PARAM.Channel.noisy_thresh * 1e-6 | ...   Noisy
        chanStd < PARAM.Channel.flat_tresh * 1e-6;          % Flat
    % Apply flags in DataFile structure
    if any(flagBad)
        chNames = strjoin({sChan.Channel(chanIdx(flagBad)).Name},',');
        b=b+1;
  
        a = chanIdx(flagBad);
        testing = zeros(1,128);
        testing(a) = 1;
        good_label = get_sleep_label(sData);

        badMask{b,1} = sData.History(3,3);
        badMask{b,2} = good_label;
        badMask{b,3} = iFile;
        badMask{b,4} = chNames;
        badMask{b,5} = a;
        badMask{b,6} = ones(length(a), length(sData.Time));
        

%       fprintf(fileID, 'Time: %s\n',sData.Comment);
%       fprintf(fileID, 'Obviously Bad Channel: %s\n',chNames);
        bst_process('CallProcess','process_channel_setbad',sFiles(iFile),[],...
            'sensortypes', chNames);
    end
end
if ~isempty(badMask)
    ObviouslyBadChannel = struct( 'time_segment',badMask(:,1),...
        'sleep_label',badMask(:,2), ...
        'flagged_epoch',badMask(:,3),'flagged_label', badMask(:,4), ...
        'flagged_index', badMask(:,5), 'binary_Mask', badMask(:,6));
end
if isempty(badMask)
    ObviouslyBadChannel = struct();
end
% Save report
bst_report('Save', resFiles);

% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
             [PARAM.currentSubject ' | DetectVeryBadChannels'],toc);