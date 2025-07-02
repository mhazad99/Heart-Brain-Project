function validatedIbiDataTable = ValidateIbiPerEpoch(ibiDataTable)
%ValidateIbiPerEpoch 

% globals
global EPOCH;

nbEpochs = ceil(ibiDataTable.TimeFromStart(end)/EPOCH.DURATION);
epochStartTime = ibiDataTable.TimeFromStart(1) + ((1:nbEpochs)-1).*EPOCH.DURATION;
epochEndTime   =  ibiDataTable.TimeFromStart(1) + (1:nbEpochs)*EPOCH.DURATION;
validIndexes = int32.empty;  
nbValidEpochs = 0;

i=1;
while i <= nbEpochs
	ibiInEpochIndex = find(ibiDataTable.TimeFromStart > epochStartTime(i) & ...
                           ibiDataTable.TimeFromStart <= epochEndTime(i));

    % isnan(ibiDataTable.RRintervals(ibiInEpochIndex))                              
    if ~isempty(ibiInEpochIndex)
        ibiInEpoch = ibiDataTable.RRintervals(ibiInEpochIndex);
        ibiSumInEpoch = sum(ibiInEpoch);
         
        if (ibiSumInEpoch >= EPOCH.MIN_EPOCH_DRURATION_FROM_IBIs && ...
            ibiSumInEpoch <= EPOCH.MAX_EPOCH_DRURATION_FROM_IBIs)
        
            nbValidEpochs = nbValidEpochs + 1;
            validIndexes = [validIndexes ibiInEpochIndex'];
        end
    
    end
    i = i + 1;
end    

validEpochsPercent = round(100.0*double(nbValidEpochs)/double(nbEpochs),1);

% Create the validated ibi time series table.
TimeFromStart = ibiDataTable.TimeFromStart(validIndexes);
DateTime = ibiDataTable.DateTime(validIndexes);
RRintervals = ibiDataTable.RRintervals(validIndexes);
HeartRates = ibiDataTable.HeartRates(validIndexes);
ParticipantID = ibiDataTable.ParticipantID(validIndexes);
MissingPercent = ibiDataTable.MissingPercent(validIndexes);
CorrectedPercent = ibiDataTable.CorrectedPercent(validIndexes); 
DataQualityFactor = ibiDataTable.DataQualityFactor(validIndexes);
ValidEpochPercent = validEpochsPercent.*ones(length(DataQualityFactor),1);
validatedIbiDataTable = table(  ParticipantID, ...
                                TimeFromStart, ...
                                DateTime , ...
                                RRintervals, ...
                                HeartRates, ...
                                MissingPercent, ...
                                CorrectedPercent, ...
                                DataQualityFactor, ...
                                ValidEpochPercent );

end % End of ValidateIbiPerEpoch

