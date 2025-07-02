function PARAM = GetSleepStages(sFiles,PARAM)
%GETSLEEPSTAGES - Read events from raw recordings and extract sleep stages
%
% SYNOPSIS: PARAM = GetSleepStages(sFiles,PARAM)
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
% Created on: 01-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if isempty(PARAM.SleepStages)
    disp('PREP>_____GETTING SLEEP STAGES')

    possibleSleepStages = {
        'NREM1','NREM2','NREM3','REM','WAKE', ...
        'N1','N2','N3','R','W'};

    % Read data
     sData = in_bst_data(sFiles.FileName,'F');
     % Find matching sleep stages names
    idx =  ismember({sData.F.events.label},possibleSleepStages);
    if any(idx)
        % Save sleep stages
        PARAM.SleepStages = {sData.F.events(idx).label};
        fprintf('\t\t%s\n',strjoin(PARAM.SleepStages,' | '))
    else
        disp('PREP> WARNING: No sleep stagesdetected!')
    end
end