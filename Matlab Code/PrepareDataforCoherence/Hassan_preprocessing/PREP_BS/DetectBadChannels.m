function [resFiles,PARAM] = DetectBadChannels(sFiles,PARAM)
% SYNOPSIS: DetectBadChannels()
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%       NOT IMPLEMENTED: COPPIETERS, 2015 for rejection based on
%                                               standard deviation
%
% See also clean_channels
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
% Created on: 02-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iEpoch = 1:length(sFiles)
    fprintf('PREP>    Processing epoch (%d/%d)\n',iEpoch,length(sFiles))
    
    % Read epoch file
    sData = in_bst_data(sFiles(iEpoch).FileName);
    % Keep EEG only
    chanMat = in_bst_channel(sFiles(iEpoch).ChannelFile,'Channel');
    eegChan = find(strcmpi({chanMat.Channel.Type},'EEG'));
    sig = sData.F(eegChan,:);
    % Create filter 0.3-35 Hz
    fs = 1 / (sData.Time(2) - sData.Time(1));
    wn = [0.3 35] / (fs/2);
    eegFilter = fir1(3,wn);
    % filter EEG
    %     sigf = filtfilt(eegFilter,1,sig);
    [sigf,~,~] = bst_bandpass_filtfilt(sig,fs,0.3,35,0,'fir');

    % Convert array to gpu array if GPU is available. corrData = [time,channels]
    if gpuDeviceCount
        corrData = gpuArray(sigf');
    else
        corrData =sigf';
    end    
    
    % Bad channel if uncorrelated with 90% of the channel: threshold: r < 0.25
    %   function modified from ref: https://github.com/sccn/clean_rawdata/blob/master/clean_channels_nolocs.m
    min_corr = 0.25;                % 0.25 works good
    ignored_quantile = 0.1;     % 0.1 works good with min_corr  = 0.25
    bad_channels = clean_channels(corrData,fs,min_corr,ignored_quantile);
    
    if any(bad_channels)
        % Get real index of bad channels
        realBadChannelIdx = eegChan(bad_channels);
        % Flag bad channels
        sData.ChannelFlag(realBadChannelIdx) = -1;
        % Save the updated epoch file
        fullFileName = bst_fullfile(sFiles(iEpoch).FileName);
        save(fullFileName, '-struct', 'sData');
        % Print message to advise which channel are flagged as bad
        msg = 'Low correlated channels: ';
        for iChan = 1:length(realBadChannelIdx)
            if iChan == 1
                msg = [msg chanMat.Channel(realBadChannelIdx(iChan)).Name];
            else
                msg = [msg ' - ' chanMat.Channel(realBadChannelIdx(iChan)).Name];
            end
        end
        fprintf(2,'PREP>    %s\n',msg);
    end
%     % Compute Pearson correlation between channels
%     r_coefs = abs(corr(corrData));
%     % Get minimum correlation per channel
%     [min_r_coefs, iMin_r] = min(r_coefs);
%     % Compute low correlation threshold: mean(r) - 4*std(r)
%     mean_r = mean(r_coefs);
%     std_r = std(r_coefs);
%     thresh = mean_r - 4*std_r;
%     % Find uncorrelated channels
%     low_r_chan = eegChan(min_r_coefs < thresh);
end