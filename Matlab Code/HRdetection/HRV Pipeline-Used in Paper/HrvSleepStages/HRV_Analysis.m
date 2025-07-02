function hrvParameters = HRV_Analysis(sleepStages, stagePeriods, inputDataTable)
%HRV_Analysis 

%% Globals
global HRV;

hrvParameters.valid = true;
%% Compute the HRV parameters for each epoch.
hrvParametersPerEpochs = ComputeHrvParametersInEpochs(sleepStages,inputDataTable);
tot_nb_epochs = length(hrvParametersPerEpochs.RMSSD);

idxWake = find(sleepStages.encoding == 0);
RMSSD = hrvParametersPerEpochs.RMSSD(idxWake);
RMSSD(isnan(RMSSD)) = [];
hrvParameters.WAKE.RMSSD = mean(RMSSD);
SDNN = hrvParametersPerEpochs.SDNN(idxWake);
SDNN(isnan(SDNN)) = [];
hrvParameters.WAKE.SDNN = mean(SDNN);
HR_AVG = hrvParametersPerEpochs.HR(idxWake);
HR_AVG(isnan(HR_AVG)) = [];
hrvParameters.WAKE.HR_AVG = mean(HR_AVG);

idxN1 = find(sleepStages.encoding == 1);
RMSSD = hrvParametersPerEpochs.RMSSD(idxN1);
RMSSD(isnan(RMSSD)) = [];
hrvParameters.N1.RMSSD = mean(RMSSD);
SDNN = hrvParametersPerEpochs.SDNN(idxN1);
SDNN(isnan(SDNN)) = [];
hrvParameters.N1.SDNN = mean(SDNN);
HR_AVG = hrvParametersPerEpochs.HR(idxN1);
HR_AVG(isnan(HR_AVG)) = [];
hrvParameters.N1.HR_AVG = mean(HR_AVG);

idxN2 = find(sleepStages.encoding == 2);
RMSSD = hrvParametersPerEpochs.RMSSD(idxN2);
RMSSD(isnan(RMSSD)) = [];
hrvParameters.N2.RMSSD = mean(RMSSD);
SDNN = hrvParametersPerEpochs.SDNN(idxN2);
SDNN(isnan(SDNN)) = [];
hrvParameters.N2.SDNN = mean(SDNN);
HR_AVG = hrvParametersPerEpochs.HR(idxN2);
HR_AVG(isnan(HR_AVG)) = [];
hrvParameters.N2.HR_AVG = mean(HR_AVG);

idxN3 = find(sleepStages.encoding == 3);
RMSSD = hrvParametersPerEpochs.RMSSD(idxN3);
RMSSD(isnan(RMSSD)) = [];
hrvParameters.N3.RMSSD = mean(RMSSD);
SDNN = hrvParametersPerEpochs.SDNN(idxN3);
SDNN(isnan(SDNN)) = [];
hrvParameters.N3.SDNN = mean(SDNN);
HR_AVG = hrvParametersPerEpochs.HR(idxN3);
HR_AVG(isnan(HR_AVG)) = [];
hrvParameters.N3.HR_AVG = mean(HR_AVG);

idxREM = find(sleepStages.encoding == 5);
RMSSD = hrvParametersPerEpochs.RMSSD(idxREM);
RMSSD(isnan(RMSSD)) = [];
hrvParameters.REM.RMSSD = mean(RMSSD);
SDNN = hrvParametersPerEpochs.SDNN(idxREM);
SDNN(isnan(SDNN)) = [];
hrvParameters.REM.SDNN = mean(SDNN);
HR_AVG = hrvParametersPerEpochs.HR(idxREM);
HR_AVG(isnan(HR_AVG)) = [];
hrvParameters.REM.HR_AVG = mean(HR_AVG);

%% HRV parameters for WAKE stages
hrvParameters.WAKEp.RMSSDN = NaN;
hrvParameters.WAKEp.SDNN = NaN;
hrvParameters.WAKEp.HR_AVG = NaN;
      
nbWakePeriods = length(stagePeriods.wake.startIdx);
% Do not include pre and post wake periods.
wakeStart = 1;
if ~isempty(stagePeriods.preWake)
    wakeStart = 2;   
end
wakeEnd = nbWakePeriods;
if ~isempty(stagePeriods.postWake)
    wakeEnd = nbWakePeriods-1;   
end

RMSSD = double.empty;
SDNN   = double.empty; 
HR_AVG  = double.empty;
for i=wakeStart:wakeEnd
    if stagePeriods.wake.endIdx(i) > tot_nb_epochs
        interval = stagePeriods.wake.startIdx(i):tot_nb_epochs;
    else    
        interval = stagePeriods.wake.startIdx(i):stagePeriods.wake.endIdx(i);
    end
    RMSSD = [RMSSD hrvParametersPerEpochs.RMSSD(interval)];
    SDNN   = [SDNN hrvParametersPerEpochs.SDNN(interval)];
    HR_AVG = [HR_AVG hrvParametersPerEpochs.HR(interval)];
end

nanIdx = isnan(RMSSD);
RMSSD(nanIdx) = [];
hrvParameters.WAKEp.RMSSD = mean(RMSSD);

nanIdx = isnan(SDNN);
SDNN(nanIdx) = [];
hrvParameters.WAKEp.SDNN = mean(SDNN);
 
nanIdx = isnan(HR_AVG);
HR_AVG(nanIdx) = [];
hrvParameters.WAKEp.HR_AVG = mean(HR_AVG);  

%%*** HRV parameters for NREM stages
nbNremPeriods = length(stagePeriods.nrem.startIdx);
hrvParameters.NREMp.RMSSD = NaN(nbNremPeriods+1,1);
hrvParameters.NREMp.SDNN   = NaN(nbNremPeriods+1,1);
hrvParameters.NREMp.HR_AVG = NaN(nbNremPeriods+1,1);

RMSSD = double.empty;
SDNN   = double.empty; 
HR_AVG  = double.empty;
for i = 1:nbNremPeriods
    if stagePeriods.nrem.endIdx(i) > tot_nb_epochs
        interval = stagePeriods.nrem.startIdx(i):tot_nb_epochs;
    else    
        interval = stagePeriods.nrem.startIdx(i):stagePeriods.nrem.endIdx(i);
    end
    % RMSSD
    data = hrvParametersPerEpochs.RMSSD(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.NREMp.RMSSD(i) = mean(data);
    RMSSD = [RMSSD data];
    % SDNN
    data = hrvParametersPerEpochs.SDNN(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.NREMp.SDNN(i) = mean(data);
    SDNN = [SDNN data];
    % HR_AVG
    data = hrvParametersPerEpochs.HR(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.NREMp.HR_AVG(i) = mean(data);
	HR_AVG = [HR_AVG data];
end
hrvParameters.NREMp.RMSSD(nbNremPeriods+1) = mean(RMSSD);
hrvParameters.NREMp.SDNN(nbNremPeriods+1) = mean(SDNN);
hrvParameters.NREMp.HR_AVG(nbNremPeriods+1) = mean(HR_AVG);

%%*** HRV parameters for REM stages
nbRemPeriods = length(stagePeriods.rem.startIdx);
hrvParameters.REMp.RMSSD = NaN(nbRemPeriods+1,1);
hrvParameters.REMp.SDNN   = NaN(nbRemPeriods+1,1);
hrvParameters.REMp.HR_AVG = NaN(nbRemPeriods+1,1);

RMSSD = double.empty;
SDNN   = double.empty; 
HR_AVG  = double.empty;

for i = 1:nbRemPeriods
    if stagePeriods.rem.endIdx(i) > tot_nb_epochs
        interval = stagePeriods.rem.startIdx(i):tot_nb_epochs;
    else    
        interval = stagePeriods.rem.startIdx(i):stagePeriods.rem.endIdx(i);
    end
    % RMSSD
    data = hrvParametersPerEpochs.RMSSD(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.REMp.RMSSD(i) = mean(data);
    RMSSD = [RMSSD data];
    % SDNN
    data = hrvParametersPerEpochs.SDNN(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.REMp.SDNN(i) = mean(data);
    SDNN = [SDNN data];
    % HR_AVG
    data = hrvParametersPerEpochs.HR(interval);
    nonValid = find(isnan(data));
    data(nonValid) = [];
    hrvParameters.REMp.HR_AVG(i) = mean(data);
	HR_AVG = [HR_AVG data];
end  
hrvParameters.REMp.RMSSD(nbRemPeriods+1) = mean(RMSSD);
hrvParameters.REMp.SDNN(nbRemPeriods+1) = mean(SDNN);
hrvParameters.REMp.HR_AVG(nbRemPeriods+1) = mean(HR_AVG);

end % End of HRV_Analysis

