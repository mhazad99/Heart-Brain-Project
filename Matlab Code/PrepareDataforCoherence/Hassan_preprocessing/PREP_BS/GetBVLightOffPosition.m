function pos = GetBVLightOffPosition(fname)
%GETBVLIGHTOFFPOSITION - pos = GetBVLightOffPosition(fname)
%
% SYNOPSIS: pos = GetBVLightOffPosition(fname)
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


% Read marker file
fId = fopen(fname);
if fId == -1
    fclose(fId);
    error('ERROR: File %s could not be opened',fname)
end
str = fileread(fname); % read entire file in a single string
fclose(fId);

% Get "lights off" marker's position
key = 'off'; % only "off" because there is often typos in "lights"
% get the value in sample without the one marked bad
idx = strfind(str,key) + length(key) + 1;
if isempty(idx)
    idx = strfind(str,'Off') + length(key) + 1;
end
pos = sscanf(str(idx:end), '%g', 1) + 1;
