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
MINIMUM_IBIS_IN_STAGES = 20;
EPOCH_DURATIONS = 30.0; % Seconds 
THRESHOLD_PERCENTAGE = 0.01;
MIN_EPOCH_DRURATION_FROM_IBIs = (1.0 - THRESHOLD_PERCENTAGE)*EPOCH_DURATIONS;
MAX_EPOCH_DRURATION_FROM_IBIs = (1.0 + THRESHOLD_PERCENTAGE)*EPOCH_DURATIONS;

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
         
    % The ReadIbiTimeSeriesFromCSV.m file is located in the ../Utils folder.
 	[ibiDataTable, validityStatus] = ReadIbiTimeSeriesFromCSV(ibiTimeSeriesFiles(i)); 
    hrvPerStages{i}.MissingPercent = validityStatus.MissingPercent; 
    hrvPerStages{i}.CorrectedPercent = validityStatus.CorrectedPercent; 
    hrvPerStages{i}.DataQualityFactor = validityStatus.DataQualityFactor;                         

    fprintf('Processing data for participant %s (%d/%d) ...\n', string(PartipantIDs(i)),i,nbParticipants);
    hrvPerStages{i}.Partipant_ID = PartipantIDs(i);
    
    hrvParametersPerEpochs = ComputeHrvParametersInEpochs(ibiDataTable);
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
    
    sleepStageScoring = sleepStages.stageType;
    nbEpochsIbi = length(hrvParametersPerEpochs.nbSamples);
    nbEpochsStages = length(sleepStageScoring);
    nbEpochs = min(nbEpochsIbi,nbEpochsStages);
    sleepStageScoring = sleepStageScoring(1:nbEpochs);
       
	%*** HRV parameters for WAKE stages
    idxForStage = find(strcmpi(sleepStageScoring,"W"));
	RMSSDN = hrvParametersPerEpochs.RMSSDN(idxForStage);
	SDNN = hrvParametersPerEpochs.SDNN(idxForStage);
	HR = hrvParametersPerEpochs.HR(idxForStage);
    RR_SUM = hrvParametersPerEpochs.RR_SUM(idxForStage);
                     
    idxForStage = find(RR_SUM >= MIN_EPOCH_DRURATION_FROM_IBIs & ...
                       RR_SUM <= MAX_EPOCH_DRURATION_FROM_IBIs & ...
                       ~isnan(RMSSDN) & ... 
                       ~isnan(SDNN) & ...
                       ~isnan(HR)); 

    if ~isempty(idxForStage)
         hrvPerStages{i}.WAKE_STAGE.RMSSDN = mean(RMSSDN(idxForStage));
         hrvPerStages{i}.WAKE_STAGE.SDNN = mean(SDNN(idxForStage));
         hrvPerStages{i}.WAKE_STAGE.HR = mean(HR(idxForStage));
         fprintf("\tWake: HR = %f, RMSSDN = %f, SDNN = %f\n", ...
                    hrvPerStages{i}.WAKE_STAGE.HR, ...
                    hrvPerStages{i}.WAKE_STAGE.RMSSDN, ...
                    hrvPerStages{i}.WAKE_STAGE.SDNN);
    else
        hrvPerStages{i}.WAKE_STAGE.RMSSDN = NaN;
        hrvPerStages{i}.WAKE_STAGE.SDNN = NaN;
        hrvPerStages{i}.WAKE_STAGE.HR = NaN;
        fprintf("\tWake: No values computed!\n");
    end    
    
    %*** HRV parameters for REM stages
    idxForStage = find(strcmpi(sleepStageScoring,"R"));
	RMSSDN = hrvParametersPerEpochs.RMSSDN(idxForStage);
	SDNN = hrvParametersPerEpochs.SDNN(idxForStage);
	HR = hrvParametersPerEpochs.HR(idxForStage);
	RR_SUM = hrvParametersPerEpochs.RR_SUM(idxForStage);
                     
    idxForStage = find(RR_SUM >= MIN_EPOCH_DRURATION_FROM_IBIs & ...
                       RR_SUM <= MAX_EPOCH_DRURATION_FROM_IBIs & ...
                       ~isnan(RMSSDN) & ... 
                       ~isnan(SDNN) & ...
                       ~isnan(HR)); 
    if ~isempty(idxForStage)
         hrvPerStages{i}.REM_STAGE.RMSSDN = mean(RMSSDN(idxForStage));
         hrvPerStages{i}.REM_STAGE.SDNN = mean(SDNN(idxForStage));
         hrvPerStages{i}.REM_STAGE.HR = mean(HR(idxForStage));
         fprintf("\tREM: HR = %f, RMSSDN = %f, SDNN = %f\n", ...
                    hrvPerStages{i}.REM_STAGE.HR, ...
                    hrvPerStages{i}.REM_STAGE.RMSSDN, ...
                    hrvPerStages{i}.REM_STAGE.SDNN);
    else
        hrvPerStages{i}.REM_STAGE.RMSSDN = NaN;
        hrvPerStages{i}.REM_STAGE.SDNN = NaN;
        hrvPerStages{i}.REM_STAGE.HR = NaN;
        fprintf("\tREM: No values computed!\n");
    end
    
    %*** HRV parameters for NREM1 stages
    idxForStage = find(strcmpi(sleepStageScoring,"N1"));
	RMSSDN = hrvParametersPerEpochs.RMSSDN(idxForStage);
	SDNN = hrvParametersPerEpochs.SDNN(idxForStage);
	HR = hrvParametersPerEpochs.HR(idxForStage);
    RR_SUM = hrvParametersPerEpochs.RR_SUM(idxForStage);
                     
    idxForStage = find(RR_SUM >= MIN_EPOCH_DRURATION_FROM_IBIs & ...
                       RR_SUM <= MAX_EPOCH_DRURATION_FROM_IBIs & ...
                       ~isnan(RMSSDN) & ... 
                       ~isnan(SDNN) & ...
                       ~isnan(HR)); 
                   
    if ~isempty(idxForStage)
         hrvPerStages{i}.NREM1_STAGE.RMSSDN = mean(RMSSDN(idxForStage));
         hrvPerStages{i}.NREM1_STAGE.SDNN = mean(SDNN(idxForStage));
         hrvPerStages{i}.NREM1_STAGE.HR = mean(HR(idxForStage));
         fprintf("\tNREM 1: HR = %f, RMSSDN = %f, SDNN = %f\n", ...
                    hrvPerStages{i}.NREM1_STAGE.HR, ...
                    hrvPerStages{i}.NREM1_STAGE.RMSSDN, ...
                    hrvPerStages{i}.NREM1_STAGE.SDNN);
    else
        hrvPerStages{i}.NREM1_STAGE.RMSSDN = NaN;
        hrvPerStages{i}.NREM1_STAGE.SDNN = NaN;
        hrvPerStages{i}.NREM1_STAGE.HR = NaN;
        fprintf("\tNREM 1: No values computed!\n");
    end     
   
    %*** HRV parameters for NREM2 stages
    idxForStage = find(strcmpi(sleepStageScoring,"N2"));
	RMSSDN = hrvParametersPerEpochs.RMSSDN(idxForStage);
	SDNN = hrvParametersPerEpochs.SDNN(idxForStage);
	HR = hrvParametersPerEpochs.HR(idxForStage);                 
    RR_SUM = hrvParametersPerEpochs.RR_SUM(idxForStage);
                     
    idxForStage = find(RR_SUM >= MIN_EPOCH_DRURATION_FROM_IBIs & ...
                       RR_SUM <= MAX_EPOCH_DRURATION_FROM_IBIs & ...
                       ~isnan(RMSSDN) & ... 
                       ~isnan(SDNN) & ...
                       ~isnan(HR));                    
    if ~isempty(idxForStage)
         hrvPerStages{i}.NREM2_STAGE.RMSSDN = mean(RMSSDN(idxForStage));
         hrvPerStages{i}.NREM2_STAGE.SDNN = mean(SDNN(idxForStage));
         hrvPerStages{i}.NREM2_STAGE.HR = mean(HR(idxForStage));
         fprintf("\tNREM 2: HR = %f, RMSSDN = %f, SDNN = %f\n", ...
                    hrvPerStages{i}.NREM2_STAGE.HR, ...
                    hrvPerStages{i}.NREM2_STAGE.RMSSDN, ...
                    hrvPerStages{i}.NREM2_STAGE.SDNN);
    else
        hrvPerStages{i}.NREM2_STAGE.RMSSDN = NaN;
        hrvPerStages{i}.NREM2_STAGE.SDNN = NaN;
        hrvPerStages{i}.NREM2_STAGE.HR = NaN;
        fprintf("\tNREM 2: No values computed!\n");
    end  

    %*** HRV parameters for NREM3 stages
    idxForStage = find(strcmpi(sleepStageScoring,"N3"));
	RMSSDN = hrvParametersPerEpochs.RMSSDN(idxForStage);
	SDNN = hrvParametersPerEpochs.SDNN(idxForStage);
	HR = hrvParametersPerEpochs.HR(idxForStage);                  
    RR_SUM = hrvParametersPerEpochs.RR_SUM(idxForStage);
                     
    idxForStage = find(RR_SUM >= MIN_EPOCH_DRURATION_FROM_IBIs & ...
                       RR_SUM <= MAX_EPOCH_DRURATION_FROM_IBIs & ...
                       ~isnan(RMSSDN) & ... 
                       ~isnan(SDNN) & ...
                       ~isnan(HR));     
    if ~isempty(idxForStage)
         hrvPerStages{i}.NREM3_STAGE.RMSSDN = mean(RMSSDN(idxForStage));
         hrvPerStages{i}.NREM3_STAGE.SDNN = mean(SDNN(idxForStage));
         hrvPerStages{i}.NREM3_STAGE.HR = mean(HR(idxForStage));
         fprintf("\tNREM 3: HR = %f, RMSSDN = %f, SDNN = %f\n", ...
                    hrvPerStages{i}.NREM3_STAGE.HR, ...
                    hrvPerStages{i}.NREM3_STAGE.RMSSDN, ...
                    hrvPerStages{i}.NREM3_STAGE.SDNN);
    else
        hrvPerStages{i}.NREM3_STAGE.RMSSDN = NaN;
        hrvPerStages{i}.NREM3_STAGE.SDNN = NaN;
        hrvPerStages{i}.NREM3_STAGE.HR = NaN;
        fprintf("NREM 3: No values computed!\n");
    end    
    
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

