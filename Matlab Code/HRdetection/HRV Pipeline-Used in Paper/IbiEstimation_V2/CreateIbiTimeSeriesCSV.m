function CreateIbiTimeSeriesCSV(outputCsvFileName,IbiResults)
%CreateIbiTimeSeriesCSV 

nbElements = length(IbiResults.RRs);
participantStrings = strings(nbElements,1);
participantStrings(:) = IbiResults.ParticipantID;

IbiDataTable = table(participantStrings, ... % Partipant ID    
                IbiResults.TimeFromStart(2:end), ... % Time from Start [s]
                string(IbiResults.RTimes(2:end,:)), ...  Date Time [dd-mmm-yyyy HH:MM:SS.FFF]
                IbiResults.RRs, ... % RR Inerval [s]
                60.0./IbiResults.RRs, ... % Heart Rate in bpm
                IbiResults.MissingPercent.*ones(nbElements,1), ...
                IbiResults.CorrectedPercent.*ones(nbElements,1), ...
                IbiResults.DataQualityFactor.*ones(nbElements,1), ...
                'VariableNames', ...
                {'ParticipantID' 'TimeFromStart' 'DateTime' 'RRintervals' 'HeartRates' 'MissingPercent' 'CorrectedPercent' 'DataQualityFactor'});

writetable(IbiDataTable,outputCsvFileName);

end % End of CreateIbiTimeSeriesCSV
 
