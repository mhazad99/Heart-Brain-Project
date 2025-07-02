%EcgPipeline% VALIDATE_EkgFiles.m
clear
close all;
clc;

addpath('../Utils/edfread');

% globals
GlobalDefs();
global FILES
global GRAPHICS

%% 1- the Data Input Folder
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\1. Depression Remission';
%currentFolder = 'R:\IMHR\Sleep Research\. Retrospective Sleep Study\0. CLINICAL DATABASE\Cardio-Dep\Final Depression'; 
currentFolder = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan';
%currentFolder = '~/imhr/HRV_pipeline/data/badResults';
[workingFolder] = uigetdir(currentFolder, 'Select the Data Input Folder');
destinationFolder = strcat(workingFolder,FILES.FILE_SEP,'MissingData');
filetype = FILES.EDF_TYPE;
%filetype = FILES.REC_TYPE;
missingTime = 900; % Seconds

fprintf('Processing files in %s folder\n', workingFolder);
if (strcmp(filetype,FILES.REC_TYPE)==1)
    allEdfFiles = dir(strcat(workingFolder,FILES.FILE_SEP,FILES.REC_TYPE));
elseif(strcmp(filetype,FILES.EDF_TYPE)==1)
    allEdfFiles = dir(strcat(workingFolder,FILES.FILE_SEP,FILES.EDF_TYPE));
end        
nbFiles = length(allEdfFiles);

for i=1:nbFiles
    fprintf('File (%d/%d): %s\n',i,nbFiles,allEdfFiles(i).name);
    completeFileName = strcat(allEdfFiles(i).folder,FILES.FILE_SEP,allEdfFiles(i).name);
    dummyTab = split(allEdfFiles(i).name,'.');
    participantID = string(dummyTab(1));    

    try    
        [headerData,rawData] = ReadEcgEdfFile(string(completeFileName));
        fprintf('\tSampling frequency: %g Hz\n',rawData.fs)
        headerLabels = string(headerData.label);
        [~,ekgChan] = find(contains(split(headerData.label),'ekg','IgnoreCase',true) | ...
                           contains(split(headerData.label),'ecg','IgnoreCase',true));
        for j=1:length(ekgChan)   
            fprintf('\tChannel #%d -> %s\n',ekgChan(j),headerLabels(ekgChan(j)))
        end    
        %missingPrct = RawEcgSignalAnalysis(rawData.fs,rawData.ekg_r);
        nbMissingSamples = missingTime*rawData.fs;
        hMissing = ones(1,nbMissingSamples);
        zerosSeq = conv(abs(rawData.ekg_r),hMissing,'same');
        idxZeros = find(zerosSeq == 0);
        endGoodData = length(rawData.ekg_r);
        if (~isempty(idxZeros))
            startMissingIdx = idxZeros(1)-nbMissingSamples/2+1;
            endGoodData = startMissingIdx-1;
            fprintf('\tMissing Data Starts at index %d (%g [hours])\n',startMissingIdx,rawData.time(startMissingIdx)/3600)    
        end    
%         if missingPrct >= 50
%             movefile(completeFileName,destinationFolder);
%             fprintf('\tMore than 50 percents of missing data!\n');
%         end

        if FILES.CREATE_CSV_FOR_EKG == true
            pStartDateTime = posixtime(rawData.startDateTime);
%             allData = [rawData.time(1:endGoodData); rawData.ekg_r(1:endGoodData)];
%             allData = [[pStartDateTime ; rawData.fs] allData]';
%             csvwrite(ekgCsvFileName, allData);
            allData = [rawData.time(1:endGoodData); rawData.ekg_r(1:endGoodData)];
            ekgCsvFileName = strcat(allEdfFiles(i).folder,FILES.FILE_SEP,participantID,'_EKG.csv');
            WriteEkgInCsvFile(ekgCsvFileName,pStartDateTime,rawData.fs,allData);
        end
        
        
        if GRAPHICS.SHOW_INPUT_SIGNAL == true
            figure(100+i);
            if FILES.CREATE_CSV_FOR_EKG == true
                rawDataFromcsv = ReadEcgCsvFile(string(ekgCsvFileName));
%                 plot(allData(2:end,1)./3600,allData(2:end,2),'b', ...
%                      rawDataFromcsv.time./3600,rawDataFromcsv.ekg_r,'r:');
                plot(rawData.time(1:endGoodData)./3600,rawData.ekg_r(1:endGoodData),'b', ...
                     rawDataFromcsv.time./3600,rawDataFromcsv.ekg_r,'r');
                legend('EDF','CSV'); 
            else    
                plot(rawData.time./3600,rawData.ekg_r);
            end
            xlabel('Time From Start [Hours]');
            ylabel('Raw EKG value [\muvolts]');
            grid on;
            pause(1);
        end
        
        clear allData
        clear rawData
        clear rawDataFromcsv
    catch ex
        fprintf(2,'Exeption in file: %s\n',allEdfFiles(i).name)
        fprintf(2,'Exception identifier: %s\n',ex.identifier)
        fprintf(2,'Exception message: %s\n',ex.message)
    end    
end    
% End of VALIDATE_EkgFiles
