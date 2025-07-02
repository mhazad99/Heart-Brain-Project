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
wakeP = 0; % Wake period counter (Pre-Wake and Post-Wake only.
nremP = 0; % NREM period counter
remP = 0; % REM period counter
   
%% Find the start of the first REM period.
idxRem = find(sleepStages.encoding == STAGE.REM_FINE_CODE);
if ~isempty(idxRem)
    firstRemPeriodStart = idxRem(1);
else
    firstRemPeriodStart = -1;
end    

%% Scenario where there is at least one REM period in the scooring file.
if firstRemPeriodStart ~= -1
    %% *** Find the last REM period.
    idxRem = find(sleepStages.encoding(nbEpochs:-1:firstRemPeriodStart) == STAGE.REM_FINE_CODE);
    if ~isempty(idxRem)
        lastRemPeriodEnd = nbEpochs - idxRem(1) + 1;
    else
        lastRemPeriodEnd = -1;
    end
    
    lastRemPeriodStart = -1;
    for i=lastRemPeriodEnd:-1:firstRemPeriodStart
        if sleepStages.encoding(i) ~= STAGE.REM_FINE_CODE
            lastRemPeriodStart = i + 1;
            break;
        end    
    end
    if lastRemPeriodStart == -1 && firstRemPeriodStart ~= -1
        lastRemPeriodStart = firstRemPeriodStart;
    end
    
    %% *** Process epochs up to the start of the first REM period
    % Find the first NREM period up to the start of the first REM period.
    idxNrem = find(sleepStages.encoding(1:firstRemPeriodStart) == 2 | ...
                   sleepStages.encoding(1:firstRemPeriodStart) == 3);              
    if ~isempty(idxNrem)
        firstNremPeriodStart = idxNrem(1);
        firstNremPeriodEnd = firstRemPeriodStart -1;
        nbNremEpochs = firstNremPeriodEnd - firstNremPeriodStart + 1;
        if nbNremEpochs >= STAGE.NREM_THRESH 
            % First NREM period
            nremP = nremP + 1; 
            nremPeriods.startIdx(nremP) = firstNremPeriodStart;
            nremPeriods.endIdx(nremP)   = firstNremPeriodEnd;
            % Pre-Wake period if firstRemPeriodStart > 1.
            % If not => file starts with a NREM stage.
            if firstNremPeriodStart > 1
                wakeP = wakeP + 1;
                wakePeriods.startIdx(wakeP) = 1;
                wakePeriods.endIdx(wakeP) = firstNremPeriodStart-1;
            end  
        else
            % Pre-Wake period if firstRemPeriodStart > 1.
            % If not => file starts with a REM stage.
            if firstRemPeriodStart > 1
                wakeP = wakeP + 1;
                wakePeriods.startIdx(wakeP) = 1;
                wakePeriods.endIdx(wakeP) = firstRemPeriodStart-1;
            end 
        end
    else
        % Pre-Wake period if firstRemPeriodStart > 1.
        % If not => file starts with a REM stage.
        if firstRemPeriodStart > 1
            wakeP = wakeP + 1;
            wakePeriods.startIdx(wakeP) = 1;
            wakePeriods.endIdx(wakeP) = firstRemPeriodStart-1;
        end   
    end    

    %% *** Process epochs up to the start of the last REM period    
    
    % Find the next NREM period up to the start of the last REM period.
    currentRemPeriodStart = firstRemPeriodStart;
    nextRemPeriodStart = currentRemPeriodStart;
    while (currentRemPeriodStart < lastRemPeriodStart)
        if currentRemPeriodStart == nextRemPeriodStart
            nremSearchStart = currentRemPeriodStart+1;
        else
            nremSearchStart = nextRemPeriodStart;
        end    
        idxNrem = find(sleepStages.encoding(nremSearchStart:lastRemPeriodStart) == 2 | ...
                       sleepStages.encoding(nremSearchStart:lastRemPeriodStart) == 3); 
        if ~isempty(idxNrem)
            % Find the next REM period up to the start of the last REM period.
            nremPeriodStart = nremSearchStart + idxNrem(1) - 1;
            remSearchStart = nremPeriodStart + 1;
            idxRem = sleepStages.encoding(remSearchStart:lastRemPeriodStart) == STAGE.REM_FINE_CODE;
            convIdxRem = find(conv(idxRem,STAGE.REM_FILTER,'valid') == STAGE.REM_THRESH);
            if ~isempty(convIdxRem)
                nextRemPeriodStart = remSearchStart + convIdxRem(1) - 1;           
                nremPeriodEnd = nextRemPeriodStart -1;
                nbNremEpochs = nremPeriodEnd - nremPeriodStart + 1;
                if nbNremEpochs >= STAGE.NREM_THRESH 
                    % NREM period
                    nremP = nremP + 1; 
                    nremPeriods.startIdx(nremP) = nremPeriodStart;
                    nremPeriods.endIdx(nremP)   = nremPeriodEnd; 
                    % REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = currentRemPeriodStart;
                    remPeriods.endIdx(remP)   = nremPeriodStart-1; 
                    currentRemPeriodStart = nextRemPeriodStart;
                end   
%                 else
%                     % Extend the current REM period
%                     remP = remP + 1; 
%                     remPeriods.startIdx(remP) = currentRemPeriodStart;
%                     remPeriods.endIdx(remP)   = nremPeriodEnd; 
%                 end    
%                 currentRemPeriodStart = nextRemPeriodStart;
            else
                % No other REM period before the last REM period.
                nremPeriodEnd = lastRemPeriodStart - 1;
                nbNremEpochs = nremPeriodEnd - nremPeriodStart + 1;
                if nbNremEpochs >= STAGE.NREM_THRESH 
                    % NREM period
                    nremP = nremP + 1; 
                    nremPeriods.startIdx(nremP) = nremPeriodStart;
                    nremPeriods.endIdx(nremP)   = nremPeriodEnd; 
                    % REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = currentRemPeriodStart;
                    remPeriods.endIdx(remP) = nremPeriodStart-1; 
                    currentRemPeriodStart = lastRemPeriodStart;
                else
                    % REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = currentRemPeriodStart;
                    remPeriods.endIdx(remP)   = lastRemPeriodStart; 
                    currentRemPeriodStart     = lastRemPeriodStart;
                end 
            end
        else
            lastRemPeriodStart = currentRemPeriodStart;
        end    
    end

    %% *** Process epochs up to the end of the scoring file.
    
    % Find the Post-Wake period
    postWakeEnd = -1;
    postWakeStart = -1;
    postWakeIdx = find(sleepStages.encoding(nbEpochs:-1:lastRemPeriodStart) == STAGE.WAKE_FINE_CODE);
    if ~isempty(postWakeIdx)
        postWakeEnd = nbEpochs - postWakeIdx(1) + 1;
        for i=postWakeEnd:-1:lastRemPeriodStart
            if sleepStages.encoding(i) ~= STAGE.WAKE_FINE_CODE
                postWakeStart = i + 1;
                break;
            end    
        end    
    end    

    % Look for a NREM period between the start of the last REM period and
    % the end of scoring file.
    lastNremStart = -1;
    lastNremEnd = -1;
	idxNrem = find(sleepStages.encoding(lastRemPeriodStart+1:nbEpochs) == 2 | ...
                   sleepStages.encoding(lastRemPeriodStart+1:nbEpochs) == 3); 
	if ~isempty(idxNrem)
        lastNremStart = lastRemPeriodStart+idxNrem(1);
        if postWakeStart > lastNremStart
            lastNremEnd = postWakeStart - 1;
            nbNremEpochs = lastNremEnd - lastNremStart + 1;
            if nbNremEpochs < STAGE.NREM_THRESH
                % Last NREM period.
                lastNremStart = -1;
                lastNremEnd = -1;
            end     
        end    
    end    
    
    if lastNremStart ~= -1
        if postWakeStart ~= -1
            if lastNremStart < postWakeStart
                % Last NREM period
                nremP = nremP + 1; 
                nremPeriods.startIdx(nremP) = lastNremStart;
                nremPeriods.endIdx(nremP)   = lastNremEnd; 
                % Post-Wake period
                wakeP = wakeP + 1;
                wakePeriods.startIdx(wakeP) = postWakeStart;
                wakePeriods.endIdx(wakeP) = nbEpochs;
            else
                % No Post-Start - end as a NREM period
                nremP = nremP + 1; 
                nremPeriods.startIdx(nremP) = lastNremStart;
                nremPeriods.endIdx(nremP)   = nbEpochs; 
            end  
            % REM period
            if ~isempty(remPeriods.endIdx) 
                if remPeriods.endIdx(remP) ~= lastRemPeriodStart
                    % Create a new REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = lastRemPeriodStart; 
                end 
                remPeriods.endIdx(remP) = lastNremStart-1;
            else
                remP = remP + 1; 
                remPeriods.startIdx(remP) = lastRemPeriodStart; 
                remPeriods.endIdx(remP) = lastNremStart-1;
            end  
         else
            % No Post-Start - end as a NREM period
            nremP = nremP + 1; 
            nremPeriods.startIdx(nremP) = lastNremStart;
            nremPeriods.endIdx(nremP)   = nbEpochs;
            % REM period
            if ~isempty(remPeriods.endIdx) 
                if remPeriods.endIdx(remP) ~= lastRemPeriodStart
                    % Create a new REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = lastRemPeriodStart; 
                end 
                remPeriods.endIdx(remP) = lastNremStart-1;
            else
                remP = remP + 1; 
                remPeriods.startIdx(remP) = lastRemPeriodStart; 
                remPeriods.endIdx(remP) = lastNremStart-1;
            end           
        end    
    else
        % No NREM and ends as Post-Wake
        if postWakeStart ~= -1
            % Post-Wake period
            wakeP = wakeP + 1;
            wakePeriods.startIdx(wakeP) = postWakeStart;
            wakePeriods.endIdx(wakeP) = nbEpochs;
            % REM period
            if ~isempty(remPeriods.endIdx) 
                if remPeriods.endIdx(remP) ~= lastRemPeriodStart
                    % Create a new REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = lastRemPeriodStart; 
                end 
                remPeriods.endIdx(remP) = postWakeStart-1;
            else
                remP = remP + 1; 
                remPeriods.startIdx(remP) = lastRemPeriodStart; 
                remPeriods.endIdx(remP) = postWakeStart-1;
            end
        else
            % No NREM and Post-Wake and ends as REM
            if ~isempty(remPeriods.endIdx) 
                if remPeriods.endIdx(remP) ~= lastRemPeriodStart
                    % Create a new REM period
                    remP = remP + 1; 
                    remPeriods.startIdx(remP) = lastRemPeriodStart; 
                end 
                remPeriods.endIdx(remP) = nbEpochs;
            else
                remP = remP + 1; 
                remPeriods.startIdx(remP) = lastRemPeriodStart; 
                remPeriods.endIdx(remP) = nbEpochs;
            end
         end    
    end    
    
else
    % No REM epoch in the scoring file.
    
    % Find the Post-Wake period
    postWakeEnd = -1;
    postWakeStart = nbEpochs;
    postWakeIdx = sleepStages.encoding(nbEpochs:-1:1) == STAGE.WAKE_FINE_CODE;
    if ~isempty(postWakeIdx)
        postWakeEnd = nbEpochs - postWakeIdx(1) + 1;
        for i=postWakeEnd-1:1
            if sleepStages.encoding(i) ~= STAGE.WAKE_FINE_CODE
                postWakeStart = i + 1;
                break;
            end    
        end    
    end
    
     % Find the first NREM period up to the start of the post-Wake or teh end of the  
     % scoring file.
    idxNrem = find(sleepStages.encoding(1:postWakeStart) == 2 | ...
                   sleepStages.encoding(1:postWakeStart) == 3);
    if ~isempty(idxNrem)
        firstNremStart = idxNrem(1);
        nbEpochs = postWakeStart - firstNremStart;
        if nbEpochs >= STAGE.NREM_THRESH
            if firstNremStart > 1
                % Pre-Wake period
                wakeP = wakeP + 1;
                wakePeriods.startIdx(wakeP) = 1;
                wakePeriods.endIdx(wakeP) = firstNremStart-1;
                % NREM period
                nremP = nremP + 1; 
                nremPeriods.startIdx(nremP) = firstNremStart;
            else
                % NREM period
                nremP = nremP + 1; 
                nremPeriods.startIdx(nremP) = 1;
            end 
            
            if postWakeStart < nbEpochs
                % Post-Wake period
                nremPeriods.endIdx(nremP) = postWakeStart-1;
                wakeP = wakeP + 1;
                wakePeriods.startIdx(wakeP) = postWakeStart;
                wakePeriods.endIdx(wakeP) = nbEpochs;
            else
                nremPeriods.endIdx(nremP) = nbEpochs;
            end 
        else
            % Pre-Wake period
            wakeP = wakeP + 1;
            wakePeriods.startIdx(wakeP) = 1;
            wakePeriods.endIdx(wakeP) = nbEpochs;
        end 
    else
        % Pre-Wake period
        wakeP = wakeP + 1;
        wakePeriods.startIdx(wakeP) = 1;
        wakePeriods.endIdx(wakeP) = nbEpochs;
    end    
end % End of if firstRemPeriodStart ~= -1

end % End of ExtractStagePeriods function
