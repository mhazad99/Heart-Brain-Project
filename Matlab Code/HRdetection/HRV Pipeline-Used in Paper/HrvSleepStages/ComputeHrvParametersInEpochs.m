function hrvParametersPerEpochs = ComputeHrvParametersInEpochs(sleepStages, ibiDataTable)
%ComputeHrvParametersInEpochs 
global EPOCH;

    nbEpochs = length(sleepStages.stageTime);

    hrvParametersPerEpochs.nbSamples = zeros(1,nbEpochs);
	hrvParametersPerEpochs.RMSSD =  nan(1,nbEpochs);
	hrvParametersPerEpochs.SDNN =  nan(1,nbEpochs);
	hrvParametersPerEpochs.HR =  nan(1,nbEpochs);
    hrvParametersPerEpochs.RR_SUM = zeros(1,nbEpochs);

    i = 1;
    while i < nbEpochs
    
        idxInEpoch = find(ibiDataTable.TimeFromStart >= sleepStages.stageTime(i) & ...
                          ibiDataTable.TimeFromStart < sleepStages.stageTime(i+1));
        if ~isempty(idxInEpoch)
            rrIntervalsInEpoch = ibiDataTable.RRintervals(idxInEpoch);
            rrIntervalsInEpoch(isnan(rrIntervalsInEpoch)) = [];
            nbSamplesInEpoch = length(rrIntervalsInEpoch);
            if (~isempty(rrIntervalsInEpoch) && nbSamplesInEpoch > 1)
                hrvParametersPerEpochs.nbSamples(i) = nbSamplesInEpoch;
                hrvParametersPerEpochs.RMSSD(i) = sqrt(sum(diff(rrIntervalsInEpoch).^2)/(double(nbSamplesInEpoch)));
                hrvParametersPerEpochs.SDNN(i) = std(rrIntervalsInEpoch);
                hrvParametersPerEpochs.HR(i) = mean(60.0./rrIntervalsInEpoch);
                hrvParametersPerEpochs.RR_SUM(i) = sum(rrIntervalsInEpoch);
            end  
        end
         
        i = i + 1;
    end % End of while i < nbEpochs
end % End of hrvParameters

