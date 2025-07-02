function [stagePeriods] = StagePeriodsFromScoring(sleepStages)
%WakeRemNremFromStages
%
% wakePeriods:   
% nremPeriods: 
% remPeriods:

%% globals
GlobalDefs_SleepStages();
global STAGE;

%% Initialization of the output parameters.
wake.startIdx = int32.empty;
wake.endIdx = int32.empty;
nrem.startIdx = int32.empty;
nrem.endIdx = int32.empty;
rem.startIdx = int32.empty;
rem.endIdx = int32.empty;
stagePeriods.preWake   = int32.empty;
stagePeriods.postWake  = int32.empty;
stagePeriods.lightsOn  = sleepStages.lightsOn;
stagePeriods.lightsOff = sleepStages.lightsOff;

%% Initialization of local variables
nbEpochs = length(sleepStages.encoding);
stagePeriods.sleepPeriods = strings(nbEpochs,1);

%% Find the last REM period
lastRemIdx = -1;
for i = nbEpochs:-1:1
    if sleepStages.encoding(i) == 5
        lastRemIdx = i;
        break;
    end   
end % End of for i = nbEpochs

%% Find the START index of last REM period
lastRemPeriodStartIdx = -1;
if lastRemIdx > 1
    for i = lastRemIdx-1:-1:1
        if sleepStages.encoding(i) ~= 5
            lastRemPeriodStartIdx = i+1;
            break;
        end   
    end    
end % End of if lastRemIdx > 1

%% Process epochs into two segments.
if lastRemPeriodStartIdx > 1
    % Epochs up to and excluding the start of the last REM period.
    seepStages1.encoding = sleepStages.encoding(1:lastRemPeriodStartIdx-1);
    [wake, nrem, rem] = ...
        ExtractStagePeriods(seepStages1, STAGE.WAKE_CODE);
    
    % Epochs from and including the start of the last REM period.
    seepStages2.encoding = sleepStages.encoding(lastRemPeriodStartIdx:end);
    [wakePeriods2, nremPeriods2, remPeriods2] = ...
        ExtractStagePeriods(seepStages2, STAGE.REM_CODE);
    
    % Merge the stage periods found in the second segment of scoring data.
    if ~isempty(wake.startIdx) && ~isempty(wakePeriods2.startIdx)
        nbPeriods = length(wake.startIdx);
        nbPeriods2 = length(wakePeriods2.startIdx);
        for i=1:nbPeriods2
            
            % This is a period that needs to be extended; 
            if wake.endIdx(nbPeriods) + 1 == ...
               wakePeriods2.startIdx(i) + lastRemPeriodStartIdx - 1
           
                wake.endIdx(nbPeriods) = ...
                    wakePeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
            
            % This is a new period to be added.
            else
                nbPeriods = nbPeriods + 1;
                wake.startIdx(nbPeriods) = ...
                    wakePeriods2.startIdx(i) + lastRemPeriodStartIdx - 1;
                wake.endIdx(nbPeriods) = ...
                    wakePeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
           end              
        end  % End of for i=1:nbPeriods2      
    end  % End of if ~isempty(wakePeriods2.startIdx) 
    
    if ~isempty(nrem.startIdx) && ~isempty(nremPeriods2.startIdx)
        nbPeriods = length(nrem.startIdx);
        nbPeriods2 = length(nremPeriods2.startIdx);
        for i=1:nbPeriods2
            
            % This is a period that needs to be extended; 
            if nrem.endIdx(nbPeriods) + 1 == ...
               nremPeriods2.startIdx(i) + lastRemPeriodStartIdx - 1
           
                nrem.endIdx(nbPeriods) = ...
                    nremPeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
            
            % This is a new period to be added.
            else
                nbPeriods = nbPeriods + 1;
                nrem.startIdx(nbPeriods) = ...
                    nremPeriods2.startIdx(i) + lastRemPeriodStartIdx - 1;
                nrem.endIdx(nbPeriods) = ...
                    nremPeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
           end              
        end  % End of for i=1:nbPeriods2
     end % End of if ~isempty(nremPeriods2.startIdx)
    
     if ~isempty(rem.startIdx) && ~isempty(remPeriods2.startIdx)
        nbPeriods = length(rem.startIdx);
        nbPeriods2 = length(remPeriods2.startIdx);
        for i=1:nbPeriods2
            
            % This is a period that needs to be extended; 
            if rem.endIdx(nbPeriods) + 1 == ...
               remPeriods2.startIdx(i) + lastRemPeriodStartIdx - 1
           
                rem.endIdx(nbPeriods) = ...
                    remPeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
            
            % This is a new period to be added.
            else
                nbPeriods = nbPeriods + 1;
                rem.startIdx(nbPeriods) = ...
                    remPeriods2.startIdx(i) + lastRemPeriodStartIdx - 1;
                rem.endIdx(nbPeriods) = ...
                    remPeriods2.endIdx(i) + lastRemPeriodStartIdx - 1;
           end              
        end  % End of for i=1:nbPeriods2
     end % if ~isempty(remPeriods2.startIdx)

%% No need to process stages into two segments (very rare and probably bad scoring data.
else
    [wake, nrem, rem] = ...
        ExtractStagePeriods(sleepStages, STAGE.WAKE_CODE);
end    

stagePeriods.wake = wake;
% Pre-wake
if ~isempty(wake.startIdx) && wake.startIdx(1) == 1
    stagePeriods.preWake = [wake.startIdx(1) wake.endIdx(1)];
end   
% Post-wake
nremLastStart = -1;
if  ~isempty(nrem.startIdx)
    nremLastStart = nrem.startIdx(end);
end
remLastStart = -1;
if  ~isempty(rem.startIdx)
    remLastStart = rem.startIdx(end);
end
if ~isempty(wake.startIdx) && ...
   wake.startIdx(end) > nremLastStart && ....
   wake.startIdx(end) > remLastStart   
	stagePeriods.postWake = [wake.startIdx(end) wake.endIdx(end)];
end    

stagePeriods.nrem = nrem;
stagePeriods.rem = rem;

% Wake periods
nbWakePer = length(wake.startIdx);
for i=1:nbWakePer
    stagePeriods.sleepPeriods(wake.startIdx(i):wake.endIdx(i)) = "WAKE";
end   
% NREM periods
nbNremPer = length(nrem.startIdx);
for i=1:nbNremPer
    nremStr = strcat("NREMp",num2str(i));
    stagePeriods.sleepPeriods(nrem.startIdx(i):nrem.endIdx(i)) = nremStr;
end   
% REM periods
nbRemPer = length(rem.startIdx);
for i=1:nbRemPer
    remStr = strcat("REMp",num2str(i));
    stagePeriods.sleepPeriods(rem.startIdx(i):rem.endIdx(i)) = remStr;
end 


end % End of StagePeriodsFromScoring Function
