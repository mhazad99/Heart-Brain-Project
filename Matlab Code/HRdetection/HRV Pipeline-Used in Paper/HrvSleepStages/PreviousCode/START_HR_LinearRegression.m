% START_HR_LinearRegression
% 
% Depression/Remission EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission
% Anxiety EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Anx\sent to medibio\Processed
% Depression EKG data (.REC):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression 
% Control EKG data (.EDF):
%   R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\0. Carrier\Dépistages_Carrier_EDF2020
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

% Globals
GlobalDefs();
global UNITS;
global HR_FIT;
global GRAPHICS;

%% 1- the Data Input Folder
%currentFolder = 'R:\IMHR\Sleep Research\Daniel\MysaPaper';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Anx\sent to medibio\Processed';
currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. Ctrls\0. Carrier\Dépistages_Carrier_EDF2020';

[workingFolder] = uigetdir(currentFolder, 'Select the Data Input Folder');
inputFolder = strcat(string(workingFolder),filesep,"IBIs");
outputFolder = strcat(string(workingFolder),filesep,"MODELS");
if ~exist(outputFolder, 'dir')
	mkdir(outputFolder)
end 
graphicsFolder = strcat(outputFolder,filesep,"GRAPHICS");
if ~exist(graphicsFolder, 'dir')
	mkdir(graphicsFolder)
end 

% Initialization of variables
ParticipantID = string.empty;
MissingPercent = int32.empty;
CorrectedPercent = int32.empty;
DataQualityFactor = int32.empty;
ValidEpochPercent = int32.empty;
hrSlopes = double.empty;
hrIntercepts = double.empty;
hrR2 = double.empty;
hrDeltaAvg = double.empty;
ibiSlopes = double.empty;
ibiIntercepts = double.empty;
ibiR2 = double.empty;
ibiDeltaAvg = double.empty;

% Read all sleep stages files (.txt).
allEntries = dir(strcat(inputFolder,filesep,'*.csv'));
nbParticipants = length(allEntries);
j = 0;
for i=1:nbParticipants
    
    dummy = split(allEntries(i).name,'.');
    fileId = string(dummy(1));    
    hrTitleText = sprintf('Linear Regression of HR Time Series\n%s', strrep(fileId,'_','-'));
    ibiTitleText = sprintf('Linear Regression of IBI Time Series\n%s', strrep(fileId,'_','-'));
    % Raw (not validated) IBI-HR Time Series.    
    if ~HR_FIT.FIT_VALIDATED
        if (~contains(allEntries(i).name,'VALIDATED','IgnoreCase',true))
            hrGraphicFileName = strcat(fileId, "_HR_LR.png");
            ibiGraphicFileName = strcat(fileId, "_IBI_LR.png");
            fprintf("\nProcessing Particicpant ID: %s (%d/%d)\n",fileId, i, nbParticipants/2);
            j = j+1;
            % Read data in a table.
            inputDataTable = readtable(strcat(string(allEntries(i).folder),filesep,string(allEntries(i).name)));
        else
            continue;
        end
    % Validated IBI-HR Time Series.      
    else
        if (contains(allEntries(i).name,'VALIDATED','IgnoreCase',true))
            hrGraphicFileName = strcat(fileId, "_VALIDATED_HR_LR.png");
            ibiGraphicFileName = strcat(fileId, "_VALIDATED_IBI_LR.png");
            fprintf("\nProcessing Particicpant ID: %s (%d/%d)\n",fileId, i/2, nbParticipants/2);
            j = j+1;
            % Read data in a table.
            inputDataTable = readtable(strcat(string(allEntries(i).folder),filesep,string(allEntries(i).name)));
        else
            continue;
        end
    end
    
    ParticipantID(j) = inputDataTable.ParticipantID(1);
    MissingPercent(j) = inputDataTable.MissingPercent(1);
    CorrectedPercent(j) = inputDataTable.CorrectedPercent(1);
    DataQualityFactor(j) = inputDataTable.DataQualityFactor(1);
    if HR_FIT.FIT_VALIDATED
        ValidEpochPercent(j) = round(inputDataTable.ValidEpochPercent(1));
    end

    %% Perform linear regression on HR time series
    dataToFit = inputDataTable.HeartRates;
    timeFromStart = inputDataTable.TimeFromStart;
    
    if HR_FIT.FILTER_TIME_SERIES == true
        stdDataToFit = std(dataToFit);
        medianDataToFit = median(dataToFit);
        lowerBound = max(medianDataToFit - 2*stdDataToFit,0);
        upperBound = medianDataToFit + 2*stdDataToFit;
        invalidIndex = find(dataToFit < lowerBound | dataToFit > upperBound); 
        timeFromStart(invalidIndex) = [];
        dataToFit(invalidIndex) = [];
    end
    
    [dataP, dataS] = polyfit(timeFromStart,dataToFit,1);
    hrSlopes(j) = dataP(1);
    hrIntercepts(j) = dataP(2);
    hrR2(j) = (dataS.normr/norm(dataToFit - mean(dataToFit)))^2;
    [dataFit, delta] = polyval(dataP,timeFromStart,dataS);
    hrDeltaAvg(j) = mean(delta);
    
    %% Display linear regression of Heart Rates Time Series
    if (GRAPHICS.SHOW_DATA_FIT_GRAPHICS == true)   
        timeFromStartInHours = timeFromStart./UNITS.SECONDS_PER_HOUR;
        
        figure(1);   

        plot(timeFromStartInHours,dataToFit,'b');
        hold on;
        Xvalues = linspace(timeFromStartInHours(1), timeFromStartInHours(end));
        plot(timeFromStartInHours,dataFit,'r-', 'LineWidth',2);
        plot(timeFromStartInHours,dataFit+2*hrDeltaAvg(j),'m--', ...
             timeFromStartInHours,dataFit-2*hrDeltaAvg(j),'m--', ...
             'LineWidth',2);
        hold off;
        xlabel('Time from Start [Hours]');
        ylabel('Heart Rate [Bpm]'); 
        xlim([0 max(timeFromStartInHours(end),Xvalues(end))+0.1])
        title(hrTitleText);
        legend('Data','Linear Fit','95% Prediction Interval');
        grid on; 

        outputPngFile = strcat(graphicsFolder, ...
                               filesep, ...
                               hrGraphicFileName, ...
                               '.png');
        saveas(gcf,outputPngFile);
        
    end % End of if (GRAPHICS.SHOW_DATA_FIT_GRAPHICS)

    %% Perform linear regression on IBI time series
    dataToFit = inputDataTable.RRintervals;
    timeFromStart = inputDataTable.TimeFromStart;
    
    if HR_FIT.FILTER_TIME_SERIES == true
        stdDataToFit = std(dataToFit);
        medianDataToFit = median(dataToFit);
        lowerBound = max(medianDataToFit - 2*stdDataToFit,0);
        upperBound = medianDataToFit + 2*stdDataToFit;
        invalidIndex = find(dataToFit < lowerBound | dataToFit > upperBound); 
        timeFromStart(invalidIndex) = [];
        dataToFit(invalidIndex) = [];
    end    
    
    
    [dataP, dataS] = polyfit(timeFromStart,dataToFit,1);
    ibiSlopes(j) = dataP(1);
    ibiIntercepts(j) = dataP(2);
    ibiR2(j) = (dataS.normr/norm(dataToFit - mean(dataToFit)))^2;
    [dataFit, delta] = polyval(dataP,timeFromStart,dataS);
    ibiDeltaAvg(j) = mean(delta);
    
    %% Display linear regression of IBI Time Series
    if (GRAPHICS.SHOW_DATA_FIT_GRAPHICS == true)   
        timeFromStartInHours = timeFromStart./UNITS.SECONDS_PER_HOUR;
        
        figure(2);       
        plot(timeFromStartInHours,dataToFit,'b');
        hold on;
        Xvalues = linspace(timeFromStartInHours(1), timeFromStartInHours(end));
        plot(timeFromStartInHours,dataFit,'r-', 'LineWidth',2);
        plot(timeFromStartInHours,dataFit+2*ibiDeltaAvg(j),'m--', ...
             timeFromStartInHours,dataFit-2*ibiDeltaAvg(j),'m--', ...
             'LineWidth',2);
        xlabel('Time from Start [Hours]');
        ylabel('IBI [Seconds]'); 
        xlim([0 max(timeFromStartInHours(end),Xvalues(end))+0.1])
        title(ibiTitleText);

        legend('Data','Linear Fit','95% Prediction Interval');
        grid on; 

        hold off;
        
        outputPngFile = strcat(graphicsFolder, ...
                               filesep, ...
                               ibiGraphicFileName, ...
                               '.png');
        saveas(gcf,outputPngFile);

        pause(1);
    end % End of if (GRAPHICS.SHOW_DATA_FIT_GRAPHICS)
end % End of for i=1:nbParticipants  


% Creating a RESULTS SUMMARY file   
currentDateTime = datetime();
currentDateTimeString = datestr(currentDateTime,'dd-mmm-yyyy_HH-MM-SS');
if ~HR_FIT.FIT_VALIDATED
        
    summaryTable = table(ParticipantID', ...
                         hrSlopes', ...
                         hrIntercepts' , ...
                         hrR2', ...
                         hrDeltaAvg', ...
                         ibiSlopes', ...
                         ibiIntercepts', ...
                         ibiR2', ...
                         ibiDeltaAvg', ...
                         MissingPercent', ...
                         CorrectedPercent', ...
                         DataQualityFactor', ...
                         'VariableNames', ...
                        {'ParticipantID' 'hrSlopes' 'hrIntercepts' 'hrR2' 'hrDeltaAvg' ...
                         'ibiSlopes' 'ibiIntercepts' 'ibiR2' 'ibiDeltaAvg' ...
                         'MissingPercent' 'CorrectedPercent' 'DataQualityFactor' });
	summaryFileName = strcat(outputFolder,filesep,'SUMMARY_',currentDateTimeString,'.csv');
else
    summaryTable = table(ParticipantID', ...
                         hrSlopes', ...
                         hrIntercepts' , ...
                         hrR2', ...
                         hrDeltaAvg', ...
                         ibiSlopes', ...
                         ibiIntercepts', ...
                         ibiR2', ...
                         ibiDeltaAvg', ...
                         MissingPercent', ...
                         CorrectedPercent', ...
                         DataQualityFactor', ...
                         ValidEpochPercent', ...
                        'VariableNames', ...
                        {'ParticipantID' 'hrSlopes' 'hrIntercepts' 'hrR2' 'hrDeltaAvg' ...
                         'ibiSlopes' 'ibiIntercepts' 'ibiR2' 'ibiDeltaAvg' ...
                         'MissingPercent' 'CorrectedPercent' 'DataQualityFactor' 'ValidEpochPercent'});
     summaryFileName = strcat(outputFolder,filesep,'SUMMARY_VALIDATED_',currentDateTimeString,'.csv');                
end
writetable(summaryTable,summaryFileName);
