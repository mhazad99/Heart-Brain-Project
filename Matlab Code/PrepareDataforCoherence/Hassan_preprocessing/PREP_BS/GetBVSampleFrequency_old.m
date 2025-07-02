function Fs = GetBVSampleFrequency(fname)
%GETBVSAMPLEFREQUENCY - Read a .vhdr file and return the sample frequency of the recording
%
% SYNOPSIS: Fs = GetBVSampleFrequency(fname)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1099231 (R2019a) Update 1 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 06-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fId = fopen(fname);
if fId == -1
    fclose(fId);
    error('ERROR: File %s could not be opened',fname)
end
str = fileread(fname); % read entire file in a single string
fclose(fId);

% Get recording sample frequency
%---------------
key = 'Sampling Rate [Hz]: ';   % pattern to look for...
idx = strfind(str,key) + length(key); % start index of character value
Fs = sscanf(str(idx:end), '%g', 1); % get the value

if isempty(idx)
    key = 'SamplingInterval=';
    idx = strfind(str,key) + length(key); % start index of character value
    Ts = sscanf(str(idx:end), '%g', 1);
    Fs = 1 / (Ts * 1e-6); % get the value
end