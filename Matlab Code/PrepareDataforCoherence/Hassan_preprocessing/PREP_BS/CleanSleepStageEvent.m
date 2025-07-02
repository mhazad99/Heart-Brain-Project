function CleanSleepStageEvent(sFiles,sleepStages)
%CLEANSLEEPSTAGEEVENT - Remove sleep stage marker of too short duration
%
% SYNOPSIS: CleanSleepStageEvent(sFiles)
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
% Created on: 27-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% REMOVE WRONG SLEEP STAGE OCCURENCES
%%%%%%%%%%%%%%%%
fprintf('PREP>\tRemoving sleep stages event of 1 sample\n')
for iFile = 1 : length(sFiles)
    if iFile == 1
        fprintf('File (%d/%d)',iFile,length(sFiles))
    else
        PrintProgress(iFile,length(sFiles))
    end
    % Read get sleep stages events
    saveEvt = false;
    sEvents = in_bst_data(sFiles(iFile).FileName,'Events');
    [~,locB] = ismember(sleepStages,{sEvents.Events.label});
    ssEvtIdx = find(locB);
    % Find and remove sleep stages event occurences with duration of 1 sample
    for iSS = 1:length(ssEvtIdx)
        iEvt = locB(ssEvtIdx(iSS));
        iOccurs= find(sEvents.Events(iEvt).times(1,:) == sEvents.Events(iEvt).times(2,:));
        
        if ~isempty(iOccurs)
            % Remove event occurrences
            sEvents.Events(iEvt).times(:,iOccurs)  = [];
            sEvents.Events(iEvt).epochs(iOccurs)   = [];
            sEvents.Events(iEvt).channels(iOccurs) = [];
            sEvents.Events(iEvt).notes(iOccurs)    = [];
            if ~isempty(sEvents.Events(iEvt).reactTimes)
                sEvents.Events(iEvt).reactTimes(iOccurs) = [];
            end
            saveEvt = true;
        end
    end
    
    if saveEvt
        bst_save(file_fullpath(sFiles(iFile).FileName), sEvents,'v6',1)
    end
end
