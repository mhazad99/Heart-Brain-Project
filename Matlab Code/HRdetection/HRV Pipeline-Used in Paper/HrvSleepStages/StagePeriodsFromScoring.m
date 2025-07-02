function [stagePeriods] = StagePeriodsFromScoring(sleepStages)
%WakeRemNremFromStages
%
% wakePeriods:   
% nremPeriods: 
% remPeriods:

%% globals

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
stagePeriods.valid = true;

%% Initialization of local variables
nbEpochs = length(sleepStages.encoding);
stagePeriods.sleepPeriods = strings(nbEpochs,1);

% Extraction of the stage periods from the individual epochs scoing.
%[wake, nrem, rem] = ExtractStagePeriods(sleepStages);
[wake, nrem, rem] = ExtractStagePeriods(sleepStages);
  
% Pre-wake
if ~isempty(wake.startIdx) && wake.startIdx(1) == 1
    stagePeriods.preWake = [wake.startIdx(1) wake.endIdx(1)];
end   
% Post-wake
if  ~isempty(wake.startIdx)
    if length(wake.startIdx) == 2
        stagePeriods.postWake = [wake.startIdx(2) wake.endIdx(2)];
    elseif wake.startIdx(1) ~= 1
        stagePeriods.postWake = [wake.startIdx(1) wake.endIdx(1)];
    end    
end

stagePeriods.nrem = nrem;
stagePeriods.rem = rem;
stagePeriods.wake = wake;

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
