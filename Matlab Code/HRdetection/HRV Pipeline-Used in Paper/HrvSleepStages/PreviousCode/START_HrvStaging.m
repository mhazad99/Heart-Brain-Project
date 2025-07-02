% START_HrvStaging
% 
% Depression/Remission EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission
% Anxiety EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Anx\sent to medibio\Processed
% Depression EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression 
% Control EKG data (.EDF):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\ALL Control Recordings
% Sleep Apnea EKG Data (.REC and .EDF):
%   R:\IMHR\Sleep Research\. CSCN Sleep Apnea Data (Najib)\EDFs
% Bipolar EKG Data:
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\EDF INPUT-OUTPUT\Bipolar Out
%   TODO: See if more cases in R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Bipolar vs Unipolar
% Bipolar Mysa (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Bipolar Disorder and HRV\EDF
%
clear all;
close all;
clc;

addpath('../Utils');

% Constants
GENERATE_RESULT_FILE = true;
DISPLAY_HRV_STAGES_GRAPH = true;

%% 1- the Data Input Folder
currentFolder = 'R:\IMHR\Sleep Research\Daniel\MysaPaper';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression';
[workingFolder] = uigetdir(currentFolder, 'Select the Data Input Folder');

% Read all sleep stages files (.txt).
allEntries = dir(strcat(workingFolder,filesep,'*.txt'));

j = 1;
PartipantIDs = string.empty;
% The following is very specific to the data file name in the folder and
% may change from project to project.
for i=1:length(allEntries)
    if (~contains(allEntries(i).name,'tag','IgnoreCase',true) && ...
         contains(allEntries(i).name,'ret','IgnoreCase',true))
        sleepStagesFile = string(strcat(allEntries(i).folder,filesep,allEntries(i).name));
        dummy = split(allEntries(i).name,'.');
        PartipantID = string(dummy(1));
        % Find the corresponding IBI time series.
        searchFolder = string(strcat(allEntries(i).folder,filesep,'IBIs'));
        ibiTimeSeriesFile = GetFileFromID(searchFolder,'*.csv',PartipantID);
        if (~isempty(ibiTimeSeriesFile) && exist(ibiTimeSeriesFile))
            PartipantIDs(j) = PartipantID;
            sleepStagesFiles(j) = sleepStagesFile;
            ibiTimeSeriesFiles(j) = ibiTimeSeriesFile;
            j = j + 1;
        else
            fprintf("\tFile %s not present\n",ibiTimeSeriesFile);
        end    
    end
end    

nbParticipants = length(PartipantIDs);
for i=1:nbParticipants
    % Get IBI Time-Series data   
    if (~exist(ibiTimeSeriesFiles(i)))
        continue;
    end   
         
 	[ibiDataTable, validityStatus] = ReadIbiTimeSeriesFromCSV(ibiTimeSeriesFiles(i)); 
    hrvPerStages{i}.MissingPercent = validityStatus.MissingPercent; 
    hrvPerStages{i}.CorrectedPercent = validityStatus.CorrectedPercent; 
    hrvPerStages{i}.DataQualityFactor = validityStatus.DataQualityFactor;                         

    fprintf('Processing data for participant %s (%d/%d) ...\n', string(PartipantIDs(i)),i,nbParticipants);
    hrvPerStages{i}.Partipant_ID = PartipantIDs(i);
    
    ComputeHrvParametersInEpochs(ibiDataTable);
    
    
    % Get sleep stages. There are two formats of sleep statges data files.
	try
        sleepStages = ReadSleepStagesFormat2(sleepStagesFiles(i), ibiDataTable.DateTime(1));   
    catch ex
        fprintf(2,"\n\tException Message: %s\n",ex.message);
        hrvPerStages{i}.WAKE_STAGE = [];
        hrvPerStages{i}.NREM1_STAGE = []; 
        hrvPerStages{i}.NREM2_STAGE = [];
        hrvPerStages{i}.NREM3_STAGE = [];
        hrvPerStages{i}.REM_STAGE = [];
        continue;
    end    
    
	%*** HRV parameters for WAKE stages
    nbStages = length(sleepStages.wakeStartTimes);
    % Get corresponding IBI data
    stageIbiData = double.empty;
    for j=1:nbStages
        stageIndex = find( ...
            ibiDataTable.DateTime >= sleepStages.wakeStartTimes(j) & ...
            ibiDataTable.DateTime <= sleepStages.wakeEndTimes(j));
        if ~isempty(stageIndex)
            stageIbiData = [stageIbiData; ibiDataTable.RRintervals(stageIndex)];
        end
    end     

    hrvPerStages{i}.WAKE_STAGE = ComputeHrvParameters(stageIbiData);
    
    %*** HRV parameters for REM stages
    nbStages = length(sleepStages.remStartTimes);
    % Get corresponding IBI data
    stageIbiData = double.empty;
    for j=1:nbStages
        stageIndex = find( ...
            ibiDataTable.DateTime >= sleepStages.remStartTimes(j) & ...
            ibiDataTable.DateTime <= sleepStages.remEndTimes(j));
        if ~isempty(stageIndex)
            stageIbiData = [stageIbiData; ibiDataTable.RRintervals(stageIndex)];
        end
    end     

    hrvPerStages{i}.REM_STAGE = ComputeHrvParameters(stageIbiData);
    
    %*** HRV parameters for NREM1 stages
    nbStages = length(sleepStages.nrem1StartTimes);
    % Get corresponding IBI data
    stageIbiData = double.empty;
    for j=1:nbStages
        stageIndex = find( ...
            ibiDataTable.DateTime >= sleepStages.nrem1StartTimes(j) & ...
            ibiDataTable.DateTime <= sleepStages.nrem1EndTimes(j));
        if ~isempty(stageIndex)
            stageIbiData = [stageIbiData; ibiDataTable.RRintervals(stageIndex)];
        end
    end     

    hrvPerStages{i}.NREM1_STAGE = ComputeHrvParameters(stageIbiData);
    
    %*** HRV parameters for NREM2 stages
    nbStages = length(sleepStages.nrem2StartTimes);
    % Get corresponding IBI data
    stageIbiData = double.empty;
    for j=1:nbStages
        stageIndex = find( ...
            ibiDataTable.DateTime >= sleepStages.nrem2StartTimes(j) & ...
            ibiDataTable.DateTime <= sleepStages.nrem2EndTimes(j));
        if ~isempty(stageIndex)
            stageIbiData = [stageIbiData; ibiDataTable.RRintervals(stageIndex)];
        end
    end     

    hrvPerStages{i}.NREM2_STAGE = ComputeHrvParameters(stageIbiData);
    
    %*** HRV parameters for NREM3 stages
    nbStages = length(sleepStages.nrem3StartTimes);
    % Get corresponding IBI data
    stageIbiData = double.empty;
    for j=1:nbStages
        stageIndex = find( ...
            ibiDataTable.DateTime >= sleepStages.nrem3StartTimes(j) & ...
            ibiDataTable.DateTime <= sleepStages.nrem3EndTimes(j));
        if ~isempty(stageIndex)
            stageIbiData = [stageIbiData; ibiDataTable.RRintervals(stageIndex)];
        end
    end     

    hrvPerStages{i}.NREM3_STAGE = ComputeHrvParameters(stageIbiData);
    
    if DISPLAY_HRV_STAGES_GRAPH == true
        figure(111);
        title(strrep(hrvPerStages{i}.Partipant_ID,'_','-'));
        yyaxis left
        plot(ibiDataTable.DateTime,ibiDataTable.HeartRates);
        ylabel('Heart Rate [Beats/min]')
        yyaxis right
        plot(sleepStages.stageDateTime,sleepStages.encoding);
        ylabel('Sleep Stage');
        xlabel('DateTime');
        grid on;
        
        outputPngFile = strcat(workingFolder, ...
                               filesep, ...
                               'IBIs', ...
                               filesep, ...
                               hrvPerStages{i}.Partipant_ID, ...
                               '_STAGING.png');
        saveas(gcf,outputPngFile);
    end    
end % End of for i=1:length(allEntries)

if GENERATE_RESULT_FILE == true
    currentDateTime = datetime();
    currentDateTimeString = datestr(currentDateTime,'dd-mmm-yyyy_HH-MM-SS');
    SummaryFileName = strcat(workingFolder,filesep,'SUMMARY_',currentDateTimeString,'.csv');
    CreateCsvResults(SummaryFileName,hrvPerStages);
end

