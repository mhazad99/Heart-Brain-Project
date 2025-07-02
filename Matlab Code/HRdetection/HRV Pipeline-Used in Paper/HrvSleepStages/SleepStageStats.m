function stageStats = SleepStageStats(sleepStages, stagePeriods, tagData)
%SleepStageStats 
%   Computes sleep stages as documented in the "Sleep Varaibles.docx"
%   document in the current folder.
%% globals
global EPOCH;
global STAGE;

stageStats.valid = true;
if ~isnan(tagData.lightsOn_epoch)
    lightsOn_epoch = tagData.lightsOn_epoch;
    stageStats.lightsOn = tagData.lightsOn_time;
else    
    lightsOn_epoch  = stagePeriods.lightsOn;
    stageStats.lightsOn = "";
end

if ~isnan(tagData.lightsOff_epoch)
    lightsOff_epoch= tagData.lightsOff_epoch;
    stageStats.lightsOff = tagData.lightsOff_time;
else    
    lightsOff_epoch  = stagePeriods.lightsOff;
    stageStats.lightsOff = "";
end

stageStats.preWake = NaN;
if ~isempty(stagePeriods.preWake)
    stageStats.preWake = stagePeriods.preWake(2)*EPOCH.MIN_PER_EPOCH;
end
stageStats.postWake = NaN;
if ~isempty(stagePeriods.postWake)
    stageStats.postWake = (stagePeriods.postWake(2)-stagePeriods.postWake(1)+1)*...
                        EPOCH.MIN_PER_EPOCH;
end

%% Time in Bed.
stageStats.TiB = NaN;
if ~isnan(lightsOff_epoch) && ~isnan(lightsOn_epoch)
	stageStats.TiB = (lightsOn_epoch - lightsOff_epoch + 1)*EPOCH.MIN_PER_EPOCH;
end

%% Sleep Onset Latency.
stageStats.SOL = NaN;
firstEpochOfSleep = NaN;
lastEpochOfSleep = NaN;
nbEpochs = length(sleepStages.epochs);
    
% First epoch of sleep.
for i=1:nbEpochs
    if sleepStages.encoding(i) ~= 0
        firstEpochOfSleep = i;
        break;
    end
end    

% Last epoch of sleep.
for i=nbEpochs:-1:1
    if sleepStages.encoding(i) ~= 0
        lastEpochOfSleep = i;
        break;
    end
end

if ~isnan(firstEpochOfSleep) && ~isnan(lightsOff_epoch)
    stageStats.SOL = (firstEpochOfSleep - lightsOff_epoch)*EPOCH.MIN_PER_EPOCH;
end     

%% Wake After Sleep Onset
stageStats.WASO = NaN;
if ~isnan(firstEpochOfSleep) && ~isnan(lastEpochOfSleep)
    idxWake = find(sleepStages.encoding(firstEpochOfSleep:lastEpochOfSleep) == 0);
    stageStats.WASO = length(idxWake)*EPOCH.MIN_PER_EPOCH; 
end

%% REM Latency.
stageStats.REM_Lat = NaN;
if ~isnan(firstEpochOfSleep) && ~isnan(lastEpochOfSleep)
    idxRem = find(sleepStages.encoding(firstEpochOfSleep:lastEpochOfSleep) == 5);
    if ~isempty(idxRem)
        stageStats.REM_Lat = (idxRem(1) - firstEpochOfSleep) * ...
                              EPOCH.MIN_PER_EPOCH;    
    end
end

%% Wake
nbWakePer = length(stagePeriods.wake.startIdx);
stageStats.WAKE_min = zeros(nbWakePer+1,1);
for i=1:nbWakePer
     stageStats.WAKE_min(i) = (stagePeriods.wake.endIdx(i) - stagePeriods.wake.startIdx(i)) * ...
                              EPOCH.MIN_PER_EPOCH;
end    
stageStats.WAKE_min(nbWakePer+1) = sum(stageStats.WAKE_min);

%% NREM sleep time
nrem1_idx = find(sleepStages.encoding == 1);
stageStats.N1_min = length(nrem1_idx)*EPOCH.MIN_PER_EPOCH;
nrem2_idx = find(sleepStages.encoding == 2);
stageStats.N2_min = length(nrem2_idx)*EPOCH.MIN_PER_EPOCH;
nrem3_idx = find(sleepStages.encoding == 3);
stageStats.N3_min = length(nrem3_idx)*EPOCH.MIN_PER_EPOCH;
stageStats.NREM_min = stageStats.N1_min + stageStats.N2_min + stageStats.N3_min;

%% REM sleep time
rem_idx = find(sleepStages.encoding == 5);
stageStats.REM_min = length(rem_idx)*EPOCH.MIN_PER_EPOCH;

%% Total Sleep Time.
stageStats.TST = stageStats.NREM_min + stageStats.REM_min;

%% Relative NREM sleep time
stageStats.N1_PC = round(100.0*double(stageStats.N1_min)/double(stageStats.TST),1);
stageStats.N2_PC = round(100.0*double(stageStats.N2_min)/double(stageStats.TST),1);
stageStats.N3_PC = round(100.0*double(stageStats.N3_min)/double(stageStats.TST),1);

%% Relative REM sleep time
stageStats.REM_PC = round(100.0*double(stageStats.REM_min)/double(stageStats.TST),1);

%% Sleep Efficiency
stageStats.SleepEfficiency = NaN;
if ~isnan(stageStats.TiB) && stageStats.TST > 0
    stageStats.SleepEfficiency = round(double(stageStats.TiB)/double(stageStats.TST),1);
end

%% NREM sleep time  per period
nbNremPer = length(stagePeriods.nrem.startIdx);
stageStats.NREMp_min = zeros(nbNremPer,1);
stageStats.NREMp_Wake_pc = zeros(nbNremPer,1);
stageStats.NREMp_N1_pc = zeros(nbNremPer,1);
stageStats.NREMp_N2_pc = zeros(nbNremPer,1);
stageStats.NREMp_N3_pc = zeros(nbNremPer,1);
stageStats.NREMp_NREM_pc = zeros(nbNremPer,1);
stageStats.NREMp_REM_pc = zeros(nbNremPer,1);
for i=1:nbNremPer
    nbEpochsInPeriod = stagePeriods.nrem.endIdx(i) - stagePeriods.nrem.startIdx(i) + 1;
    stageStats.NREMp_min(i) = nbEpochsInPeriod * EPOCH.MIN_PER_EPOCH;
    % Pourcentage of epoch scored as WAKE in each NREM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i)) == STAGE.WAKE_FINE_CODE));
    stageStats.NREMp_Wake_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as NREM 1 in each NREM period.
    nbStagesNrem = 0;
    nbStages = length(find(sleepStages.encoding(stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i)) == 1));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.NREMp_N1_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as NREM 2 in each NREM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i)) == 2));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.NREMp_N2_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1); 
    % Pourcentage of epoch scored as NREM 3 and 4 in each NREM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i)) == 3));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.NREMp_N3_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as NREM (1,2,3, or 4) in each NREM period.
    stageStats.NREMp_NREM_pc(i) = round(100.0*double(nbStagesNrem)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as REM in each NREM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i)) == STAGE.REM_FINE_CODE));
    stageStats.NREMp_REM_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
end    
stageStats.NREMp_Tot_min = sum(stageStats.NREMp_min);

%% REM sleep time
nbRemPer = length(stagePeriods.rem.startIdx);
stageStats.REMp_min =  zeros(nbRemPer,1);
stageStats.REMp_REM_pc = zeros(nbRemPer,1);
stageStats.REMp_Wake_pc = zeros(nbRemPer,1);
stageStats.REMp_N1_pc = zeros(nbRemPer,1);
stageStats.REMp_N2_pc = zeros(nbRemPer,1);
stageStats.REMp_N3_pc = zeros(nbRemPer,1);
stageStats.REMp_NREM_pc = zeros(nbRemPer,1);
for i=1:nbRemPer
    nbEpochsInPeriod = stagePeriods.rem.endIdx(i) - stagePeriods.rem.startIdx(i) + 1;
    stageStats.REMp_min(i) = nbEpochsInPeriod * EPOCH.MIN_PER_EPOCH;
    % Pourcentage of epoch scored as REM in each REM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i)) == STAGE.REM_FINE_CODE));
    stageStats.REMp_REM_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as WAKE in each REM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i)) == STAGE.WAKE_FINE_CODE));
    stageStats.REMp_Wake_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);
    % Pourcentage of epoch scored as NREM 1 in each REM period.
    nbStagesNrem = 0;
    nbStages = length(find(sleepStages.encoding(stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i)) == 1));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.REMp_N1_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1);    
    % Pourcentage of epoch scored as NREM 2 in each REM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i)) == 2));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.REMp_N2_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1); 
    % Pourcentage of epoch scored as NREM 3 and 4 in each REM period.
    nbStages = length(find(sleepStages.encoding(stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i)) == 3));
    nbStagesNrem = nbStagesNrem + nbStages;
    stageStats.REMp_N3_pc(i) = round(100.0*double(nbStages)/double(nbEpochsInPeriod),1); 
    % Pourcentage of epoch scored as NREM (1,2,3, or 4) in each REM period.
    stageStats.REMp_NREM_pc(i) = round(100.0*double(nbStagesNrem)/double(nbEpochsInPeriod),1);
end
stageStats.REMp_Tot_min = sum(stageStats.REMp_min);

%% Total Sleep Time.
stageStats.TSTp = stageStats.NREMp_Tot_min + stageStats.REMp_Tot_min;

%% Relative NREM sleep time
stageStats.NREMp_Tot_pc = round(100.0*double(stageStats.NREMp_Tot_min)/double(stageStats.TSTp),1);

%% Relative REM sleep time
stageStats.REMp_Tot_pc = round(100.0*double(stageStats.REMp_Tot_min)/double(stageStats.TSTp),1);

end % End of SleepStageStats function

