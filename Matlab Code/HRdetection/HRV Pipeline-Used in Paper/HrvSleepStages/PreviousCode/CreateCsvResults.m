function CreateCsvResults(baseFileName,hrvResults)
%CREATECSVRESULTS 

header = 'Filename,HR_Wake,HR_NREM1,HR_NREM2,HR_NREM3,HR_REM,SDNN_Wake,SDNN_NREM1,SDNN_NREM2,SDNN_NREM3,SDNN_REM,RMSSD_Wake,RMSSD_NREM1,RMSSD_NREM2,RMSSD_NREM3,RMSSD_REM,';

nbResults = length(hrvResults);
rowOfData = double.empty;
rowsOfCsvFile = strings(1,nbResults);
nbField = 15;
for i=1:nbResults
    rowsOfCsvFile(i) = strcat(hrvResults{i}.Partipant_ID, ',');
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
end    
 
csvFileName = strcat(baseFileName,'_MAPS','_HRV','.csv');
fid = fopen(csvFileName,'w');
fprintf(fid,'%s\n',header);
for i=1:nbResults
    fprintf(fid,'%s\n', rowsOfCsvFile(i));
end    
fclose(fid);

end % End of CreateCsvResults
 
