% START_EkgPipeline
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
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression 
%       \Batch1 ...
%       \Batch2 ...\2A
%       \Batch2 ...\2B
%       \Batch3 ...\3A
%       \Batch3 ...\3B
%       \Batch4
%       \Batch5
%       \Batch6
%       \Batch7
%       \Batch8
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

function [Res] = START_EkgPipeline(currentFolder)
clear
close all;
clc;

% globals
GlobalDefs();
global FILES

%% 1- the Data Input Folder
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\0. Carrier\Dépistages_Carrier_EDF2020';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Bipolar Disorder and HRV\EDF';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission\EKG';
%currentFolder = '/Volumes/USB DRIVE/ecg_data/tag_files';
%currentFolder = 'J:/Dépistages_Carrier_EDF2020';
currentFolder = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Temporary folder';

%[workingFolder] = uigetdir(currentFolder, 'Select the Data Input Folder');
workingFolder = currentFolder;
% Uncomment it if you want to have a dialogue box about scoring file format
% scoringFileFormat = questdlg('Sleep Scoring File Format:', ...
%                              'Sleep Scroring File', ...
%                              'Ancestry Format', ...
%                              'Latest Format', ...
%                              'Ancestry Format');
scoringFileFormat = 'Ancestry Format';

recFiles = dir(strcat(workingFolder,FILES.FILE_SEP,FILES.REC_TYPE));
edfFiles = dir(strcat(workingFolder,FILES.FILE_SEP,FILES.EDF_TYPE));
if ~isempty(recFiles)
    [Res] = EcgPipeline(workingFolder,datetime.empty,FILES.REC_TYPE, scoringFileFormat);
elseif ~isempty(edfFiles)
    [Res] = EcgPipeline(workingFolder,datetime.empty,FILES.EDF_TYPE, scoringFileFormat); 
else
    fprintf(2, "\nNo EDF files (*.edf or *.rec) found in the %s folder.\n", workingFolder);
end

HR = 60./Res.RRs;
Time = Res.TimeFromStart;
output_dir = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\HR Analysis\Test\Depression Cases';
% Create the full output file path
output_file = fullfile(output_dir, ['HR_' Res.ParticipantID '.mat']);
save(output_file, "HR", "Time");

end

% End of START_EkgPipeline

