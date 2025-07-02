function [wakePeriods, nremPeriods, remPeriods] = ExtractStagePeriods(sleepStages)
%WakeRemNremFromStages
%
% wakePeriods:   
% nremPeriods: 
% remPeriods:

%% globals
GlobalDefs_SleepStages();
global STAGE;

%% Initialization of the output parameters.
wakePeriods.startIdx = int32.empty; % Pre-wake and post-wake only.
wakePeriods.endIdx = int32.empty;   % Post-wake and post-wake only.
nremPeriods.startIdx = int32.empty;
nremPeriods.endIdx = int32.empty;
remPeriods.startIdx = int32.empty;
remPeriods.endIdx = int32.empty;

%% Initialization of local variables
nbEpochs = length(sleepStages.encoding);
nremP = 0; % NREM period counter
nremCounter = 0;
remP = 0; % REM period counter
remCounter = 0;
remPeriodThreshold = STAGE.REM_THRESH_FIRST_PERIOD;

%% Find the start of the first NREM period.
idxNrem = sleepStages.encoding == 2 | ...
          sleepStages.encoding == 3;
convIdxNrem = find(conv(idxNrem,STAGE.NREM_FILTER,'valid') == STAGE.NREM_THRESH);
firstNremPeriodStart = convIdxNrem(1);

%% Find the start of the first REM period.
idxRem =  find(sleepStages.encoding == STAGE.REM_FINE_CODE);
firstRemPeriodStart = idxRem(1);


%% First NREM period before first REM period (usual scenario)
wakeP = 1; % Pre-wake period
if firstNremPeriodStart < firstRemPeriodStart
    wakePeriods.startIdx(wakeP) = 1;
    wakePeriods.endIdx(wakeP)   = firstNremPeriodStart-1;
    currentState = STAGE.NREM_CODE;
    startIdx = firstNremPeriodStart;
else
% First REM period before first NREM period (rare case)
    wakePeriods.startIdx(wakeP) = 1;
    wakePeriods.endIdx(wakeP)   = firstRemPeriodStart-1;
    currentState = STAGE.REM_CODE;
    startIdx = firstRemPeriodStart;
end
previousState = currentState;

%% Find the last REM period
idxRem = find(sleepStages.encoding(nbEpochs:-1:1)== STAGE.REM_FINE_CODE);
lastRemPeriodEndIdx = nbEpochs - idxRem(1) + 1;
if lastRemPeriodEndIdx > nbEpochs
    lastRemPeriodEndIdx = nbEpochs;
end

%% Find the START index of last REM period
lastRemPeriodStartIdx = -1;
if lastRemPeriodEndIdx > 1
    for i = lastRemPeriodEndIdx-1:-1:1
        if sleepStages.encoding(i) ~= 5
            lastRemPeriodStartIdx = i+1;
            break;
        end   
    end    
end % End of if lastRemIdx > 1

endIdx = lastRemPeriodStartIdx-1;
if endIdx > nbEpochs
    endIdx = nbEpochs
end    

%% Process all epochs up to the last REM period.
i = startIdx;
while i <= endIdx
    
    switch sleepStages.encoding(i)
        %% Wake
        case 0
            if currentState == STAGE.NREM_CODE  
            	nremCounter = nremCounter + 1;
                remCounter = 0;
            elseif currentState == STAGE.REM_CODE
                remCounter = remCounter + 1;
                nremCounter = 0;
            end
              
        %% NREM
        case 1
            if currentState == STAGE.NREM_CODE  
            	nremCounter = nremCounter + 1;
                remCounter = 0;
            else
                nremCounter = 0;
            end
            
        case {2,3,4}
            nremCounter = nremCounter + 1;
            if currentState == STAGE.NREM_CODE
                remCounter = 0;
            end
            
            if nremCounter == STAGE.NREM_THRESH
                % New NREM period
                nremP = nremP + 1;
                nremPeriods.startIdx(nremP) = i - STAGE.NREM_THRESH + 1;
                nremPeriods.endIdx(nremP) = 0;
             
                if currentState == STAGE.REM_CODE && remP > 0
                    remPeriods.endIdx(remP) = i - STAGE.NREM_THRESH;
                    remPeriodThreshold = STAGE.REM_THRESH;
                end
                previousState = currentState;
                currentState = STAGE.NREM_CODE;
            end
            
        %% REM
        case 5
            remCounter = remCounter + 1;
            if currentState == STAGE.REM_CODE
                nremCounter = 0;
            end
             
            if remCounter == remPeriodThreshold    
                % New REM period
                remP = remP + 1;
                remPeriods.startIdx(remP) = i - remPeriodThreshold + 1;
                remPeriods.endIdx(remP) = 0;

                % Previous state was NREM state
                if currentState == STAGE.NREM_CODE && nremP > 0
                    nremPeriods.endIdx(nremP) = i - remPeriodThreshold;    
                end
                previousState = currentState;
                currentState = STAGE.REM_CODE;
            end   
    end
    i = i + 1; % Next epochs
    
end % while i <= endIdx

%% Close the last stage period
% Previous state was WAKE state
if currentState == STAGE.NREM_CODE
    nremPeriods.endIdx(nremP) = endIdx; 
    % Also add the last REM period previously identified.
    remP = remP + 1;
    remPeriods.startIdx(remP) = lastRemPeriodStartIdx;
    remPeriods.endIdx(remP)   = lastRemPeriodEndIdx;
elseif currentState == STAGE.REM_CODE
    % This is the last REM period that continues with what was identified as the last
    % REM period.
    remPeriods.endIdx(remP) = lastRemPeriodEndIdx; 
end

%% Processing of the remaining data (endIdx+1:nbEpochs)
% Find the start of the next NREM period.
idxNrem = sleepStages.encoding(endIdx+1:nbEpochs) == 1 | ...
          sleepStages.encoding(endIdx+1:nbEpochs) == 2 | ...
          sleepStages.encoding(endIdx+1:nbEpochs) == 3;
convIdxNrem = find(conv(idxNrem,STAGE.NREM_FILTER,'valid') == STAGE.NREM_THRESH);
if ~isempty(convIdxNrem)
    nextNremPeriodStart = endIdx + convIdxNrem(1);
else
    nextNremPeriodStart = -1;
end    

% Find the last wake (post-wake) end index.
lastWakePeriodEnd = -1;
for i=nbEpochs:-1:endIdx+1  
    if sleepStages.encoding(i) == STAGE.WAKE_FINE_CODE
        lastWakePeriodEnd = i;
        break;
    end
end   
% Find the last wake (post-wake) start index.
lastWakePeriodStart = -1;
if lastWakePeriodEnd ~= -1
    for i=lastWakePeriodEnd:-1:endIdx+1
        if sleepStages.encoding(i) ~= STAGE.WAKE_FINE_CODE
            lastWakePeriodStart = i+1;
            break;
        end
    end    
end

if nextNremPeriodStart == -1 
    if lastWakePeriodStart == -1
        % Extend the last REM period to the end.
        remPeriods.endIdx(remP) = nbEpochs;
    else
        % The scoring end in the WAKE state (NO NREM period after the last REM period).
        remPeriods.endIdx(remP) = lastWakePeriodStart-1;
        % The is a Post-Wake period.
        wakeP = wakeP + 1;
        wakePeriods.startIdx(wakeP) = lastWakePeriodStart;
        wakePeriods.endIdx(wakeP)   = nbEpochs;
    end
else
    if lastWakePeriodStart == -1 || lastWakePeriodStart < nextNremPeriodStart
        % The scoring end in the NREM state (No Post-Wake after the last REM period).
        remPeriods.endIdx(remP) = nextNremPeriodStart-1;
        % The is the last NREM period.
        nremP = nremP + 1;
        nremPeriods.startIdx(nremP) = nextNremPeriodStart;
        nremPeriods.endIdx(nremP)   = nbEpochs;
    else
        % There is a NREM and a Post-Wake periods after the last REM period.
        % nextNremPeriodStart < nextWakePeriodStart 
        remPeriods.endIdx(remP) = nextNremPeriodStart-1;
        nremP = nremP + 1;
        % The is the last NREM period.
        nremPeriods.startIdx(nremP) = nextNremPeriodStart;
        nremPeriods.endIdx(nremP)   = lastWakePeriodStart-1;
        % The is a Post-Wake period.
        wakeP = wakeP + 1;
        wakePeriods.startIdx(wakeP) = lastWakePeriodStart;
        wakePeriods.endIdx(wakeP)   = nbEpochs;
    end    
end    

end % End of ExtractStagePeriods function

