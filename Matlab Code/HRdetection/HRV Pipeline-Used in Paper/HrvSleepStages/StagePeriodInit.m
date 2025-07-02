function stagePeriodStruct = StagePeriodInit()
%StagePeriodInit 
	stagePeriodStruct.preWake = int32.empty;
    stagePeriodStruct.postWake = int32.empty;
    stagePeriodStruct.lightsOn = NaN;
    stagePeriodStruct.lightsOff = NaN;
    stagePeriodStruct.sleepPeriods = string.empty;
    stagePeriodStruct.nrem.startIdx = int32.empty;
    stagePeriodStruct.nrem.endIdx = int32.empty;
    stagePeriodStruct.rem.startIdx = int32.empty;
    stagePeriodStruct.rem.endIdx = int32.empty;
    stagePeriodStruct.wake.startIdx = int32.empty;
    stagePeriodStruct.wake.endIdx = int32.empty;
    stagePeriodStruct.valid = false;
end

