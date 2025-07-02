function hrLinearFit = HR_LinearRegression(stagePeriods, inputDataTable)
%HR_LinearRegression 

%% Globals
global HR_FIT;
global EPOCH;

%% Perform linear regression fit of HR time series over the complete night.
dataToFit = inputDataTable.HeartRates;
timeFromStart = inputDataTable.TimeFromStart;
isNanIdx = isnan(dataToFit);
dataToFit(isNanIdx) = [];
timeFromStart(isNanIdx) = [];
hrLinearFit.valid = true;
    
if HR_FIT.FILTER_TIME_SERIES == true
    stdDataToFit = std(dataToFit);
    medianDataToFit = median(dataToFit);
    lowerBound = max(medianDataToFit - 2*stdDataToFit,0);
    upperBound = medianDataToFit + 2*stdDataToFit;
    invalidIndex = find(dataToFit < lowerBound | dataToFit > upperBound); 
    timeFromStart(invalidIndex) = [];
    dataToFit(invalidIndex) = [];
end
    
[dataP, dataS] = polyfit(timeFromStart,dataToFit,1);
hrLinearFit.slope = dataP(1);
hrLinearFit.intercept = dataP(2);
hrLinearFit.R2 = (dataS.normr/norm(dataToFit - mean(dataToFit)))^2;
[dataFit, delta] = polyval(dataP,timeFromStart,dataS);
hrLinearFit.delta = mean(delta);

%% Perform linear regression fit of HR time series for each NREM Period.
hrLinearFit.nrem.slope = double.empty;
hrLinearFit.nrem.intercept = double.empty;
hrLinearFit.nrem.R2 = double.empty;
hrLinearFit.nrem.delta = double.empty;

nbNremPeriods = length(stagePeriods.nrem.startIdx);
for i=1:nbNremPeriods
    startTime = stagePeriods.nrem.startIdx(i)*EPOCH.DURATION;
    endTime = stagePeriods.nrem.endIdx(i)*EPOCH.DURATION;
    periodIdx = find (timeFromStart >= startTime & ...
                      timeFromStart <= endTime);
    periodData = dataToFit(periodIdx);
    periodTime = timeFromStart(periodIdx);
    [dataP, dataS] = polyfit(periodTime,periodData,1);
    hrLinearFit.nrem.slope(i) = dataP(1);
    hrLinearFit.nrem.intercept(i) = dataP(2);
    hrLinearFit.nrem.R2(i) = (dataS.normr/norm(periodData - mean(periodData)))^2;
    [dataFit, delta] = polyval(dataP,periodTime,dataS);
    hrLinearFit.nrem.delta(i) = mean(delta);
end    


%% Perform linear regression fit of HR time series for each REM Period.
hrLinearFit.rem.slope = double.empty;
hrLinearFit.rem.intercept = double.empty;
hrLinearFit.rem.R2 = double.empty;
hrLinearFit.rem.delta = double.empty;

nbRemPeriods = length(stagePeriods.rem.startIdx);
for i=1:nbRemPeriods
    startTime = stagePeriods.rem.startIdx(i)*EPOCH.DURATION;
    endTime = stagePeriods.rem.endIdx(i)*EPOCH.DURATION;
    periodIdx = find (timeFromStart >= startTime & ...
                      timeFromStart <= endTime);
    periodData = dataToFit(periodIdx);
    periodTime = timeFromStart(periodIdx);
    [dataP, dataS] = polyfit(periodTime,periodData,1);
    hrLinearFit.rem.slope(i) = dataP(1);
    hrLinearFit.rem.intercept(i) = dataP(2);
    hrLinearFit.rem.R2(i) = (dataS.normr/norm(periodData - mean(periodData)))^2;
    [dataFit, delta] = polyval(dataP,periodTime,dataS);
    hrLinearFit.rem.delta(i) = mean(delta);
end    


end % End of HR_LinearRegression
