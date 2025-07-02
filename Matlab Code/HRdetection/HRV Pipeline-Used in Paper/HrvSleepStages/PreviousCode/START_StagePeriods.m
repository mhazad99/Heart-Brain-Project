% Test_StagePeriods
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


%% 1- The Data Input Folder
currentFolder = pwd;
[workingFolder] = uigetdir(currentFolder, 'Select the Data Input Folder');

% Create output folder 
outputFolder = strcat(string(workingFolder),filesep,"STAGE_PERIODS");
if ~exist(outputFolder, 'dir')
	mkdir(outputFolder)
end 

% Read all sleep stages files (.txt).
allEntries = dir(strcat(workingFolder,filesep,'*.txt'));

PartipantIDs = string.empty;
nbScoringFiles = length(allEntries);
%stagePeriods(nbScoringFiles,1) = struct;
%stageStats(nbScoringFiles,1) = struct;
for i=1:nbScoringFiles

	sleepStagesFile = string(strcat(allEntries(i).folder,filesep,allEntries(i).name));
	dummy = split(allEntries(i).name,'.');
	PartipantIDs(i) = string(dummy(1));
	fprintf("Participant: %s (%d/%d)\n",PartipantIDs(i),i,nbScoringFiles);
    try
        %% Read the sleep stages from input scoring file.
        sleepStages = ReadSleepStages(sleepStagesFile); 
        %% Classify stages into stage periods. 
        stagePeriods(i) =  StagePeriodsFromScoring(sleepStages);
        %% Compute statistics from stage periods
        stageStats(i) = SleepStageStats(stagePeriods(i));
        
        %% Save scoring stages and stage periods to csv file.
%         filename = strcat(outputFolder,filesep,PartipantID,'_scoring.csv');
%         particpants = strings(nbEpochs,1);
%         scoringTable = table(particpants, ...
%                              sleepStages.epochs', ...
%                              sleepStages.stageTime', ...
%                              sleepStages.encoding', ...
%                              sleepStages.stageType' , ...
%                              sleepStages.stagePeriods', ...
%                              'VariableNames', ...
%                             {'ParticipantID' 'Epoch' 'TimeFromStart' 'StageEncoding' ...
%                              'StageScore' 'StagePeriod' });
%         writetable(scoringTable,filename);
    catch ex
        fprintf(2,"\tCannot process scoring file %s\n",allEntries(i).name);
        fprintf(2,"\tException message: %s\n",getReport(ex));
        continue
    end
end  % End of for i=1:nbScoringFiles  

