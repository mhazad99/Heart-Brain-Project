function [srate, dt, recFileName, mrkFileName] = f_GetMFFSampleFrequency(fullFileName)
%F_GETMFFSAMPLEFREQUENCY - Read MFF file info to get sampling rate, recording 
%   start time and recording file name.
%
% SYNOPSIS: [srate, dt, recFileName] = f_GetMFFSampleFrequency(fullFileName)
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
%   MATLAB ver.: 9.6.0.1011450 (R2019a) Prerelease on
%    Linux 4.15.0-72-generic #81~16.04.1-Ubuntu SMP Tue Nov 26 16:34:21 UTC 2019 
%              x86_64
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medecine (CARSM)
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 18-Dec-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Get recording name and build new marker name from it
recFileName = fullFileName;
[fPath,fName,~] = fileparts(fullFileName);
mrkFileName = fullfile(fPath,[fName '.vmrk']);

% Get recording time
try
    hdr = mff_importinfo(recFileName);
catch
    addpath('mffmatlabio')
    try
        hdr = mff_importinfo(recFileName);
    catch
        disp('RL2BV> Select mffmatlabio toolbox directory')
        mffToolboxDir = uigetdir([],'Select mffmatlabio toolbox directory');
        addpath(mffToolboxDir);
        hdr = mff_importinfo(recFileName);
    end
end
dt = datetime(hdr.recordtimematlab,'Format','yyyyMMddHHmmssSSSSSS','ConvertFrom','datenum'); %hdr.recordtimematlab si mff_importinfo

% Get Sampling rate
[~, ~, frequencies, ~] = mff_importsignal(recFileName);
srate = frequencies(1);

