function CreateCsvResults(baseFileName,hrvResults)
%CREATECSVRESULTS 

header = 'Filename,Missing Data [Percent],IBI Corrected [Percent],Overall Quality [Percent],HR_Wake [beats/min],HR_NREM1 [beats/min],HR_NREM2 [beats/min],HR_NREM3 [beats/min],HR_REM [beats/min],SDNN_Wake [s],SDNN_NREM1 [s],SDNN_NREM2 [s],SDNN_NREM3 [s],SDNN_REM [s],RMSSD_Wake [s],RMSSD_NREM1 [s],RMSSD_NREM2 [s],RMSSD_NREM3 [s],RMSSD_REM [s],';

nbResults = length(hrvResults);
rowOfData = double.empty;
rowsOfCsvFile = strings(1,nbResults);
nbField = 15;
for i=1:nbResults
    rowsOfCsvFile(i) = strcat(hrvResults{i}.Partipant_ID, ',', ...
                        num2str(hrvResults{i}.MissingPercent), ',', ...
                        num2str(hrvResults{i}.CorrectedPercent), ',', ...
                        num2str(round(hrvResults{i}.DataQualityFactor)), ',');
                    
    if (~isempty(hrvResults{i}.WAKE_STAGE) && ...
        ~isempty(hrvResults{i}.NREM1_STAGE)&& ... 
        ~isempty(hrvResults{i}.NREM2_STAGE)&& ...
        ~isempty(hrvResults{i}.NREM3_STAGE)&& ...
        ~isempty(hrvResults{i}.REM_STAGE))
 
        rowOfData(1) = hrvResults{i}.WAKE_STAGE.HR;
        rowOfData(2) = hrvResults{i}.NREM1_STAGE.HR;
        rowOfData(3) = hrvResults{i}.NREM2_STAGE.HR;
        rowOfData(4) = hrvResults{i}.NREM3_STAGE.HR;
        rowOfData(5) = hrvResults{i}.REM_STAGE.HR;

        rowOfData(6)  = hrvResults{i}.WAKE_STAGE.SDNN;
        rowOfData(7)  = hrvResults{i}.NREM1_STAGE.SDNN;
        rowOfData(8)  = hrvResults{i}.NREM2_STAGE.SDNN;
        rowOfData(9)  = hrvResults{i}.NREM3_STAGE.SDNN;
        rowOfData(10) = hrvResults{i}.REM_STAGE.SDNN;

        rowOfData(11) = hrvResults{i}.WAKE_STAGE.RMSSDN;
        rowOfData(12) = hrvResults{i}.NREM1_STAGE.RMSSDN;
        rowOfData(13) = hrvResults{i}.NREM2_STAGE.RMSSDN;
        rowOfData(14) = hrvResults{i}.NREM3_STAGE.RMSSDN;
        rowOfData(15) = hrvResults{i}.REM_STAGE.RMSSDN;

        for j=1:nbField
            if isnan(rowOfData(j))
                rowsOfCsvFile(i) = strcat(rowsOfCsvFile(i),',');
            else
                rowsOfCsvFile(i) = strcat(rowsOfCsvFile(i),num2str(rowOfData(j)),',');
            end
        end    
    else
        rowsOfCsvFile(i) = strcat(rowsOfCsvFile(i),',,,,,,,,,,,,,,,');
    end    
end    
 
csvFileName = strcat(baseFileName,'_HRV','.csv');
fid = fopen(csvFileName,'w');
fprintf(fid,'%s\n',header);
for i=1:nbResults
    fprintf(fid,'%s\n', rowsOfCsvFile(i));
end    
fclose(fid);

end % End of CreateCsvResults
 
