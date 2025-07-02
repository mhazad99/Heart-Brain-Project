function [preWake] = PreWakeFromStages(sleepStages)
%PREANDPOSTWAKEFROMSTAGES
%
% Pre-wake:  Time from start corresponding to the first non-wake epoch.
% Post-wake: Time from start corresponding to the last non-wake epoch.

EPOCH_DURATION = 30.0; % Seconds
NON_WAKE_THRESH = 15.0; % Minutes
NB_NON_WAKE_THRESH = round(NON_WAKE_THRESH*60.0/EPOCH_DURATION);

nbSleepStages = length(sleepStages.stageTime);

%% Pre-wake
nbNonWake = 0;
for i=1:nbSleepStages
    if (strcmpi(sleepStages.stageType(i), "W"))
        nbNonWake = 0;
    else
        nbNonWake = nbNonWake + 1;
        if (nbNonWake >= NB_NON_WAKE_THRESH)
            preWake.endIdx = i - NB_PRE_WAKE_THRESH;
            preWake.endTime = (preWake.endIdx-1)*EPOCH_DURATION;
            break;
        end
    end    
end

end % End of PreWakeFromStages function

