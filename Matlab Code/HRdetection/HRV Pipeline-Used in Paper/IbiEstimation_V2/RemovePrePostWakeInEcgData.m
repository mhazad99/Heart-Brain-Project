function updatedRawData = RemovePrePostWakeInEcgData(rawData, scoringFileFullPathName, scoringFileFormat)
% RemovePrePostWakeInEcgData 
global EPOCH

addpath('../HrvSleepStages');

if scoringFileFormat == "Ancestry Format"
    sleepStages = ReadSleepStages(scoringFileFullPathName); 
else
    sleepStages = ReadSleepStagesNewFormat(scoringFileFullPathName); 
end
stagePeriods =  StagePeriodsFromScoring(sleepStages);

updatedRawData.startDateTime = rawData.startDateTime;
updatedRawData.startDate = rawData.startDate;
updatedRawData.startTime = rawData.startTime;
updatedRawData.fs = rawData.fs;
updatedRawData.isDifferential = rawData.isDifferential;

afterPreWake = 0.0;
if ~isempty(stagePeriods.preWake)
    afterPreWake = (stagePeriods.preWake(end)+1)*EPOCH.DURATION;
end

beforePostWake = rawData.time(end);
if ~isempty(stagePeriods.postWake)
    beforePostWake = (stagePeriods.postWake(1)-1)*EPOCH.DURATION;
end    

workingDataIdx = find(rawData.time >= afterPreWake & ...
                      rawData.time <= beforePostWake);
if ~isempty(workingDataIdx) && afterPreWake ~= beforePostWake
    updatedRawData.ekg_r = rawData.ekg_r(workingDataIdx);
    updatedRawData.time = rawData.time(workingDataIdx);
    updatedRawData.timeOffset = rawData.time(workingDataIdx(1));
else
    updatedRawData.ekg_r = rawData.ekg_r;
    updatedRawData.time = rawData.time;
    updatedRawData.timeOffset = rawData.timeOffset;
end

end % End of RemovePrePostWakeInEcgData

