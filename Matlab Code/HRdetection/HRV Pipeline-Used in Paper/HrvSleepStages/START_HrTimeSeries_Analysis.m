function START_HrTimeSeries_Analysis()
% START_HrTimeSeries_Analysis 
%
% Control EKG data (.EDF):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\ALL Control Recordings
%       \AZ ->  No scoring file
%       \CFS -> No scoring file
%       \PLMs -> No scoring file (both edf and rec files?)
%       \REJ -> No scoring file
%       \RET -> No scoring file
%       \StartWithNumbers -> No scoring file
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\0. Carrier\Dépistages_Carrier_EDF2020
%       \N0_Milaad_2020
%       \N0_Somana_2020
%       \Opossom
% Depression/Remission EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission\EKG
% Anxiety EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Anx\sent to medibio\EKG
% Depression EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression\EKG 
% Sleep Apnea EKG Data (.REC and .EDF) but no scoring file:
%   R:\IMHR\Sleep Research\. CSCN Sleep Apnea Data (Najib)\EDFs 
%       \1_28_De_Id - No Scoring file
%       \2_39_De_Id - No Scoring file
%       \Recordings 1 - No Scoring file
%       \Recordings 2 - No Scoring file
%       \Recordings 3 - No Scoring file
%       \Recordings 4 - No Scoring file
%       \Recordings 5 - No Scoring file
%       \Recordings 6 - No Scoring file
%   R:\IMHR\Sleep Research\. CSCN Sleep Apnea Data (Najib)\Jan03_2020 
%       \1_28_De_Id - No Scoring file
%       \2_39_De_Id - No Scoring file
%       \mybox-selected page 1 - No Scoring file
%       \mybox-selected page 2 - No Scoring file
%       \mybox-selected page 3 - No Scoring file
%       \mybox-selected-page 4 - No Scoring file
%       \mybox-selected-page 5 - No Scoring file
% Bipolar Mysa (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Bipolar Disorder and HRV\EDF
% Mysa Paper:
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression and HRV in Sleep\0. EDF Files\EKG1 - EKG2\Depression
%
clear all;
clc;

% globals
GlobalDefs_SleepStages();
global EXPORT
global GRAPHICS

addpath('../Utils');

%% Initialisation
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\0. Carrier\Dépistages_Carrier_EDF2020';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression\EKG';
%currentFolder = "R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Anx\sent to medibio\EKG";
currentFolder = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan";
%currentFolder = "R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Bipolar Disorder and HRV\EDF";
%currentFolder = "J:/Dépistages_Carrier_EDF2020";
%currentFolder = '/Volumes/USB DRIVE/ecg_data/tag_files';

[workingFolder] = uigetdir(currentFolder, 'Select the Input Data Folder');
% Data input folders.
ibiDataFolder = strcat(workingFolder,filesep, "IBIs");
stageScoringDataFolder = workingFolder;
% Data output folders
validatedIbiDataFolder = strcat(ibiDataFolder,filesep, "Validated");
maxNbWakePeriods = 0;
maxNbNremPeriods = 0;
maxNbRemPeriods  = 0;

% Select the sleep scoring file format: 
scoringFileFormat = questdlg('Sleep Scoring File Format:', ...
                             'Sleep Scroring File', ...
                             'Ancestry Format', ...
                             'Latest Format', ...
                             'Ancestry Format');

%
processTagFiles = false;
% Reads lights on and lights off from tag file
tagFiles = questdlg('Search in TAG files?', 'Lights Off/Lights On', 'Yes', 'No', 'Yes');
if contains(tagFiles,'Yes')
    processTagFiles = true;
end

if EXPORT.EXECUTION_LOG_FILE == true
    currentDateStr = datestr(datetime);
    currentDateStr = strrep(currentDateStr,':','_');
    
    logFileName = strcat(workingFolder,filesep,'HR_Analysis_',currentDateStr,"*.log");
    diary logFileName;
end    

%% Get all IBI time series file names.
allIbiFiles = dir(strcat(ibiDataFolder,filesep,'*.csv'));
nbIbiDataFiles = length(allIbiFiles);
ParticipantIDs = string.empty;
for i=1:nbIbiDataFiles
	sleepStages = SleepStagesInit();
    stagePeriods(i) = StagePeriodInit();
    tagData(i) = TagDataInit();
    stageStats(i) = StageStatsInit();
    hrLinearFit(i) = HrLinearFitInit();
    hrvParameters(i) = HrvParametersInit();
%% 0- Extract the particpant ID from the input file name.
    dummy = strsplit(string(allIbiFiles(i).name),".csv");
    ParticipantIDs(i) = string(dummy(1));
    fprintf("\nPROCESSING HR Time Series for Particicpant ID : %s (%d/%d)\n", ParticipantIDs(i), i,nbIbiDataFiles);
    
%% 1- Read-in IBI Time Series data.
    fprintf("\tREADING HR Time Series ...\n");
    ibiFullFileName = strcat(ibiDataFolder, filesep, string(allIbiFiles(i).name));
    ibiDataTable = readtable(ibiFullFileName);
    
%% 2- Perform validation of IBI Time Series data on a per epoch basis.
    % Validation of the IBI Time Series
    validatedIbiDataTable = ValidateIbiPerEpoch(ibiDataTable);
    if isempty(validatedIbiDataTable)
        fprintf(2,"\tIBI Time Series for particpant ID %s is not valid ... skipping to next particpant.\n", ...
                ParticipantIDs(i));
        continue;    
    end
    fprintf("\tVALIDATING HR Time Series ...\n");
    fprintf('\t\tPercentage of valid epochs: %d\n',round(validatedIbiDataTable.ValidEpochPercent(1)));
    
    if EXPORT.VALIDATED_IBI_TIME_SERIES == true
        % Export validated IBI Time Series data.
        if ~exist(validatedIbiDataFolder,'dir')
            mkdir(validatedIbiDataFolder)
        end
            
        validatedIbisFileName = strcat(validatedIbiDataFolder, ...
                                        filesep, ...
                                        ParticipantIDs(i),...
                                        "_VALIDATED.csv");
        writetable(validatedIbiDataTable,validatedIbisFileName);
    end
    
    if GRAPHICS.VALIDATED_IBI_TIME_SERIES == true
        SHOW_IbiTimeSeries(validatedIbiDataFolder, validatedIbiDataTable);
    end

    dataQuality(i).MissingPercent = validatedIbiDataTable.MissingPercent(1);
    dataQuality(i).CorrectedPercent = validatedIbiDataTable.CorrectedPercent(1); 
    dataQuality(i).DataQualityFactor = validatedIbiDataTable.DataQualityFactor(1);
    dataQuality(i).ValidEpochPercent = validatedIbiDataTable.ValidEpochPercent(1);
    
%% 3- Read-in the corresponding sleep-stage scoring data file.
    fprintf("\tREADING Sleep Stages scoring data ...\n");
    scoringFileName = FindScoringFileFromId(stageScoringDataFolder, ParticipantIDs(i));
    scoringFileName = strcat(stageScoringDataFolder,filesep,scoringFileName);
    % Read the sleep stages from input scoring file.
    if ~isempty(scoringFileName)
        if scoringFileFormat == "Ancestry Format"
            sleepStages = ReadSleepStages(scoringFileName); 
        else
            sleepStages = ReadSleepStagesNewFormat(scoringFileName); 
        end 
    else
        fprintf(2,"\tScoring file named %s not found ... skipping to next particpant.\n", ...
            strcat(ParticipantIDs(i),".txt"));
        continue;
    end    
%% 3.b- Read sleep period from tag file
    if processTagFiles == true
        tagFileName = FindTagFileFromId(stageScoringDataFolder, ParticipantIDs(i));
        tagFileName = strcat(stageScoringDataFolder,filesep,tagFileName);
        fprintf("\tReading TAG file data ...\n");
        % Read the sleep stages from input scoring file.
        if ~isempty(tagFileName)
            tagData(i) = ReadTagFILE(tagFileName);
        else
            fprintf(2,"\tTAG file for participant %s not found.\n", ParticipantIDs(i));
        end  
    end
    
%% 4- Extract sleep-stage periods.
    nbNrem = find(sleepStages.encoding == 1 | ...
                  sleepStages.encoding == 2 | ... 
                  sleepStages.encoding == 3);
    if ~isempty(nbNrem)
        fprintf("\tEXTRACTING Sleep Stages' Periods ...\n");
        stagePeriods(i) =  StagePeriodsFromScoring(sleepStages);
        if length(stagePeriods(i).wake.startIdx) > maxNbWakePeriods
            maxNbWakePeriods = length(stagePeriods(i).wake.startIdx);       
        end

        if length(stagePeriods(i).nrem.startIdx) > maxNbNremPeriods
            maxNbNremPeriods  = length(stagePeriods(i).nrem.startIdx);
        end

        if length(stagePeriods(i).rem.startIdx) > maxNbRemPeriods
            maxNbRemPeriods  = length(stagePeriods(i).rem.startIdx);
        end
    else
        fprintf(2,"\tNon-valid data in scoring file named %s  ... skipping to next particpant.\n", ...
            strcat(ParticipantIDs(i),".txt"));
        continue;
    end    
      
%% 5- Compute sleep-stage statistics.
    fprintf("\tCOMPUTING Sleep Variables ...\n");
    stageStats(i) = SleepStageStats(sleepStages,stagePeriods(i),tagData(i));
   
%% 6- Perform linear regression analysis of validated HR time-series data.
    fprintf("\tPERFORMING Linear Fit of HR Time Series ...\n");
    hrLinearFit(i) = HR_LinearRegression(stagePeriods(i), validatedIbiDataTable);

%% 7- Compute HRV time-domain parameters on validated HR time-series data.
    fprintf("\tCOMPUTING HRV parameters on HR Time Series ...\n");
    hrvParameters(i) = HRV_Analysis(sleepStages, stagePeriods(i), validatedIbiDataTable);

end % End of for i=1:nbIbiDataFiles    

if EXPORT.EXECUTION_LOG_FILE == true
    diary off;
end

%% 8- Write results table to summary CSV data file.
maxPeriods.wake = maxNbWakePeriods;
maxPeriods.nrem = maxNbNremPeriods;
maxPeriods.rem  = maxNbRemPeriods;
WriteAnalysisSummary( workingFolder, ...
                      ParticipantIDs, ...
                      stagePeriods, ...
                      maxPeriods, ...
                      stageStats, ...
                      hrLinearFit, ...
                      hrvParameters, ...
                      dataQuality);
                  
% End of START_HrTimeSeries_Analysis

