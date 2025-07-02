function sleepStages = AskSleepStages()
%ASKSLEEPSTAGES - Ask the user to select one or more sleep stages from a list
%
% SYNOPSIS: sleepStages = AskSleepStages()
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

sleepStages = [];

stagesList = {'N1','N2','N3','R','W'};

[stageIdx,tf] = listdlg('ListString',stagesList);
if ~tf
    fprintf('User cancelled sleep stage selection. Exit\n')
    return
end
sleepStages = stagesList(stageIdx);
    
    