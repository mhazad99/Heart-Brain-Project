% START_ValidateIbiPerEpoch
clear;
clc;

addpath('../Utils');
%currentFolder = pwd;
currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression';
[dataFolder] = uigetdir(currentFolder, 'Select the IBIs Data Input Folder');
dataFolder = strcat(dataFolder,filesep,"IBIs");

allDataFiles = dir(strcat(dataFolder,filesep,'*.csv'));
nbDataFiles = length(allDataFiles);
for i=1:nbDataFiles
    if (contains(allDataFiles(i).name, "VALIDATED") || contains(allDataFiles(i).name, "MEDIAN"))
        continue;
    end
    
    dummy = strsplit(string(allDataFiles(i).name),".csv");
    fileId = string(dummy(1));
    fprintf("\nProcessing Particicpant ID: %s\n",fileId);
    fullFileName = strcat(dataFolder, filesep, string(allDataFiles(i).name));
    % Read the data in the working test file.
    %[ibiDataTable, validityStatus] = ReadIbiTimeSeriesFromCSV(fullFileName); 
    ibiDataTable = readtable(fullFileName);
    % Validation of the IPB Time Series
    [preValidationStats, postValidationStats] = ValidateIbiPerEpoch(ibiDataTable);
    
    fprintf('File: %s -> Pourcentage of valid epochs: %d (%d/%d)\n', ...
            fileId, round(postValidationStats.validEpochsPercent),i,nbDataFiles);
       
    % Display of the Heart Rate Time Series
    figure(1);
    subplot(2,1,1)
    plot(ibiDataTable.TimeFromStart./3600, ibiDataTable.HeartRates);
    xlabel('Time from Start [hours]');
    ylabel('Heart Rate [bpm]');
    grid on;
    title('Heart Rate Time Series');
    subplot(2,1,2)
    timeFromStart = ibiDataTable.TimeFromStart(postValidationStats.validIndexes)./3600;
    heartRates = ibiDataTable.HeartRates(postValidationStats.validIndexes);
    plot(timeFromStart, heartRates);
    xlabel('Time from Start [hours]');
    ylabel('Validated Heart Rate [bpm]');
    grid on;
    title('Validated Heart Rate Time Series');
    outputPngFile = strcat(dataFolder, ...
                           filesep, ...
                           fileId, ...
                           '_VALIDATED_HR.png');
	saveas(gcf,outputPngFile);
    
    % Display of the IBI Time Series
    figure(2);
    subplot(2,1,1)
    plot(ibiDataTable.TimeFromStart./3600, ibiDataTable.RRintervals);
    xlabel('Time from Start [hours]');
    ylabel('Heart Rate [bpm]');
    grid on;
    title('IBI Time Series');
    subplot(2,1,2)
    timeFromStart = ibiDataTable.TimeFromStart(postValidationStats.validIndexes)./3600;
    RRintervals = ibiDataTable.RRintervals(postValidationStats.validIndexes);
    plot(timeFromStart, RRintervals);
    xlabel('Time from Start [hours]');
    ylabel('Validated Heart Rate [bpm]');
    grid on;
    title('Validated IBI Time Series');
    
     outputPngFile = strcat(dataFolder, ...
                           filesep, ...
                           fileId, ...
                           '_VALIDATED_IBIs.png');
	saveas(gcf,outputPngFile);
    
    % Export validated IBI Time series into a CSV file.
    outputCsvFileName = strcat(dataFolder,filesep,fileId,"_VALIDATED.csv");
	TimeFromStart = ibiDataTable.TimeFromStart(postValidationStats.validIndexes);
    DateTime = ibiDataTable.DateTime(postValidationStats.validIndexes);
    RRintervals = ibiDataTable.RRintervals(postValidationStats.validIndexes);
    HeartRates = ibiDataTable.HeartRates(postValidationStats.validIndexes);
    ParticipantID = ibiDataTable.ParticipantID(postValidationStats.validIndexes);
    MissingPercent = ibiDataTable.MissingPercent(postValidationStats.validIndexes);
    CorrectedPercent = ibiDataTable.CorrectedPercent(postValidationStats.validIndexes); 
    DataQualityFactor = ibiDataTable.DataQualityFactor(postValidationStats.validIndexes);
    ValidEpochPercent = postValidationStats.validEpochsPercent.*ones(length(DataQualityFactor),1);
    outputTable = table(ParticipantID, ...
                        TimeFromStart, ...
                        DateTime , ...
                        RRintervals, ...
                        HeartRates, ...
                        MissingPercent, ...
                        CorrectedPercent, ...
                        DataQualityFactor, ...
                        ValidEpochPercent);
    writetable(outputTable,outputCsvFileName);
end