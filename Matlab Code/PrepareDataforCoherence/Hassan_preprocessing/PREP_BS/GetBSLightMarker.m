function nightLimits = GetBSLightMarker(sFile)
%GETBSLIGHTMARKER - Read latency of 'light off' and 'light on' marker from 
%   brainstorm data file structure
%
% SYNOPSIS: [nightStart, nightEnd] = GetBSLightMarker(sFile)
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
% Created on: 23-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nightLimits = [];

% Read data structure
sData = in_bst_data(sFile.FileName,'F');
% Get marker index of begining and end of the night
lightEvt = contains({sData.F.events.label},{'light','lights','ligths','ligth','lgihts'},'IgnoreCase',true);
idxOff = lightEvt & contains({sData.F.events.label},'off','IgnoreCase',true);
idxOn = lightEvt & contains({sData.F.events.label},'on','IgnoreCase',true);

% Get marker latencies in seconds
if any(idxOff)
    % Real night start
    nightLimits = sData.F.events(idxOff).times(1);
else
    % Use begining of recording as night start
    disp('PREP> WARNING: LIGHTS OFF marker not found. Use begining of recording as Night Start.')
    nightLimits = sData.F.prop.times(1);
end
if any(idxOn)
    % Real night end
    nightLimits = [nightLimits sData.F.events(idxOn).times'];
else
    % Use end of recording as night end
    disp('PREP> WARNING: LIGHTS ON marker not found. Use end of recording as Night End.')
    nightLimits = [nightLimits sData.F.prop.times(2)];
end

% if markers are extended event, use only onset of each one
if size(nightLimits,1) == 2
    nightLimits = nightLimits(1,:);
end

