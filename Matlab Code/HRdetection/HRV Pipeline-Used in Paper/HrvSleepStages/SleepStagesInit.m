function sleepStatesStruct = SleepStagesInit()
%StagePeriodInit 
	sleepStatesStruct.epochs = int32.empty;
    sleepStatesStruct.stageType = string.empty;
    sleepStatesStruct.encoding = int32.empty;
    sleepStatesStruct.stageTime = double.empty;
    sleepStatesStruct.lightsOff = NaN;
    sleepStatesStruct.lightsOn = NaN;
    sleepStatesStruct.valid = false;   
end
