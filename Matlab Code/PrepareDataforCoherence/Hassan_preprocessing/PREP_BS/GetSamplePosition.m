function pos = GetSamplePosition(startTime, evtTime,Fs)
%GETSAMPLEPOSITION - Calculate number of samples between startTime and evtTime 
%   at sample frequency Fs
%
% SYNOPSIS: pos = GetSamplePosition(startTime, evtTime,Fs)
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
% Created on: 16-May-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% SECTION NAME HERE
%%%%%%%%%%%%%%%%

t = datetime(evt{i,2},'InputFormat','HH:mm:ss.SSS');
t.Day = dt.Day; t.Month = dt.Month; t.Year = dt.Year; % use recording date
nbrSecondFromStart = seconds(t-dt);


    