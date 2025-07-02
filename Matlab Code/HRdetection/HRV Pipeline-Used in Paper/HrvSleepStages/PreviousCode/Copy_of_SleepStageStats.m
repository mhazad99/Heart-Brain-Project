function stageStats = SleepStageStats(stagePeriods)
%SleepStageStats 
%   Computes sleep stages as documented in the "Sleep Varaibles.docx"
%   document in the current folder.
%% globals
GlobalDefs_SleepStages();
global EPOCH;

stageStats.lightsOff = stagePeriods.lightsOff;
stageStats.lightsOn  = stagePeriods.lightsOn;
%% Time in Bed.
stageStats.TiB = NaN;
if ~isnan(stageStats.lightsOff) && ~isnan(stageStats.lightsOn)
	stageStats.TiB = (stageStats.lightsOn - stageStats.lightsOff)*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;
end

%% Sleep Onset Latency.
stageStats.SOL = NaN;
firstEpochOfSleep = NaN;
if ~isnan(stageStats.lightsOff)
    
    % First epoch of sleep.
    if ~isempty(stageStats.nrem.startIdx)
        if ~isempty(stageStats.rem.startIdx)
            firstEpochOfSleep = min(stageStats.nrem.startIdx(1),stageStats.rem.startIdx(1));
        else
            firstEpochOfSleep = stageStats.nrem.startIdx(1);
        end    
    else
        if ~isempty(stageStats.rem.startIdx)
            firstEpochOfSleep = stageStats.rem.startIdx(1);
        end
    end   
    
    if ~isnan(firstEpochOfSleep) 
        stageStats.SOL = (firstEpochOfSleep - stageStats.lightsOff)*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;
    end  
end    

%% Wake After Sleep Onset
stageStats.WASO = NaN;
if ~isempty(stagePeriods.postWake) && ~isNaN(firstEpochOfSleep)
    stageStats.WASO = (stagePeriods.postWak(2) - firstEpochOfSleep)*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;
end

%% REM Latency.
stageStats.REM_Lat = NaN;
if ~isnan(firstEpochOfSleep) && ~isempty(stageStats.rem.startIdx)
	stageStats.REM_Lat = (firstEpochOfSleep - stageStats.rem.startIdx(1))*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;    
end

%% NREM sleep time
stageStats.NREM_min = 0;
nbNremPer = length(stageStats.nrem.startIdx);
for i=1:nbNremPer
    stageStats.NREM_min = stageStats.NREM_min + ...
        (stageStats.nrem.endIdx(i) - stageStats.nrem.startIdx(i));
end    
stageStats.NREM_min = stageStats.NREM_min*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;

%% REM sleep time
stageStats.REM_min = 0;
nbRemPer = length(stageStats.rem.startIdx);
for i=1:nbRemPer
    stageStats.REM_min = stageStats.REM_min + ...
        (stageStats.rem.endIdx(i) - stageStats.rem.startIdx(i));
end
stageStats.REM_min = stageStats.REM_min*EPOCH.DURATION*EPOCH.EPOCHS_PER_MIN;

%% Total Sleep Time.
stageStats.TST = stageStats.NREM_min + stageStats.REM_min;

%% Relative NREM sleep time
stageStats.NREM_PC = round(100.0*stageStats.NREM_min/stageStats.TST,1);

%% Relative REM sleep time
stageStats.REM_PC = round(100.0*stageStats.REM_min/stageStats.TST,1)

%% Sleep Efficiency
stageStats.SleepEfficiency = NaN;
if ~isnan(stageStats.TiB) && stageStats.TST > 0
    stageStats.SleepEfficiency = round(stageStats.TiB/stageStats.TST,1);
end

end % End of SleepStageStats function

