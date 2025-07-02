function [Res] = EcgPipeline(folderPath, startDateTime, filetype, scoringFileFormat)
% EcgPipeline
% ECG - Pipeline - Processing all EDF files selected by the user.
%
% Pan/Hamilton-Tompkins algorithm.
% Wessel et al. Ibi Time Series adaptive filtering.
%
% Process all edf files in the folder received as a parameter.
%
% Start datetime or recording can also be passed as an argument when not
% available in EDF file.
%

% globals
GlobalDefs();
global FILES
global GRAPHICS
global ECG_WAVEFORM
global PRE_PROCESSING
global PEAK_DETECTION
global POST_PROCESSING  
addpath('../Utils/edfread');
addpath('../HrvSleepStages');
GlobalDefs_SleepStages();

fprintf('Processing files in %s folder\n', folderPath);

ibisFolder = strcat(folderPath, FILES.FILE_SEP, 'IBIs');
if ~exist(ibisFolder, 'dir')
    mkdir(ibisFolder)
end
    
if (strcmp(filetype,FILES.REC_TYPE)==true)
    allEdfFiles = dir(strcat(folderPath,FILES.FILE_SEP,FILES.REC_TYPE));
elseif(strcmp(filetype,FILES.EDF_TYPE)==true)
    allEdfFiles = dir(strcat(folderPath,FILES.FILE_SEP,FILES.EDF_TYPE));
elseif(strcmp(filetype,FILES.CSV_TYPE)==true)
    allEdfFiles = dir(strcat(folderPath,FILES.FILE_SEP,'*.csv'));    
end        
nbFiles = length(allEdfFiles);

if FILES.CREATE_LOG_FILE
    currentDateStr = datestr(datetime);
    currentDateStr = strrep(currentDateStr,':','_');

    logFileName = strcat(folderPath,FILES.FILE_SEP,'EcgPipeline_',currentDateStr,FILES.LOG_TYPE);
    logFileHandle = fopen(logFileName,'w');
    fprintf(logFileHandle,'Data folder: %s\n',folderPath);
end    

for i=1:nbFiles

    % TEMPORARY: do not read the .rec files starting with the _ character.
%     if (contains(allEdfFiles(i).name,"_") && strcmp(filetype,FILES.REC_TYPE)==true)
%         continue;
%     end
    
    completeFileName = strcat(allEdfFiles(i).folder,FILES.FILE_SEP,allEdfFiles(i).name);
    fprintf('\nReading %s file (%d/%d) ...\n', allEdfFiles(i).name, i, nbFiles);
    if FILES.CREATE_LOG_FILE
        fprintf(logFileHandle,'File: %s\n\n',allEdfFiles(i).name);
    end 
    
	try
        if (strcmp(filetype,FILES.REC_TYPE)==1 || ...
            strcmp(filetype,FILES.EDF_TYPE)==1)
            rawData = ReadEcgInputFile(string(completeFileName));
        else
            rawData = ReadEcgCsvFile(string(completeFileName));
        end   
        
        if FILES.CREATE_LOG_FILE
            if rawData.isDifferential
                fprintf(logFileHandle,'\tDIFFERENTIAL EKG channel\n');
            else
                fprintf(logFileHandle,'\tSINGLE EKG channel\n');
            end    
        end 
    catch ex
        fprintf(2,'Exeption reading file: %s\n',allEdfFiles(i).name)
        fprintf(2,'Exception identifier: %s\n',ex.identifier)
        fprintf(2,'Exception message: %s\n',ex.message)
        continue;
    end 
	
    maxTimeFromStart = rawData.time(end); % seconds
    rawData.timeOffset = 0;
    if ECG_WAVEFORM.REMOVE_PRE_POST_WAKE == true
        % Remove the pre-wake and post-wake data segments based on the
        % information from the sleep stages scoring file.
  
        stagingFolder = folderPath;
        dummy = strsplit(allEdfFiles(i).name,".");
        scoringFileName = strcat(string(dummy(1)), ".txt");
        fullScoringFileName = strcat(stagingFolder, filesep, scoringFileName);
        scoringFileFound = false;
        if exist(fullScoringFileName, 'file')
            scoringFileFound = true;            
        else   
            id = string(regexp(allEdfFiles(i).name,"RET_\d{4}",'match','ignorecase'));
            scoringFileName = FindScoringFileFromId(stagingFolder, id);
            fullScoringFileName = strcat(stagingFolder, filesep, scoringFileName);
            if ~isempty(scoringFileName) && exist(fullScoringFileName, 'file')
                scoringFileFound = true;   
            else
                id = string(regexp(allEdfFiles(i).name,"MP_\d{3}",'match','ignorecase'));
                scoringFileName = FindScoringFileFromId(stagingFolder, id);
                fullScoringFileName = strcat(stagingFolder, filesep, scoringFileName);
                if ~isempty(scoringFileName) && exist(fullScoringFileName, 'file')
                    scoringFileFound = true; 
                end   
            end
        end    
   
        if scoringFileFound == true
             % File exists then proceed.
             try
                rawData = RemovePrePostWakeInEcgData(rawData, ...
                                                     fullScoringFileName, ...
                                                     scoringFileFormat);
             catch ex
                fprintf(2,"\tCouldn't read scoring file %s!\n",scoringFileName); 
                fprintf(2,"\t%s\n",ex.message); 
                continue;
             end    
        else
             % File does not exist resume with raw ecg data.
             fprintf(2,"\tCorresponding scoring file not found!\n");
             continue;
        end
    end
    
    if (rawData.time(end) - rawData.time(1)) < ECG_WAVEFORM.MINIMUM_DURATION
        fprintf(2,"\tRecording of %f seconds has been DISCARTED!\n", (rawData.time(end) - rawData.time(1)));
        fprintf(logFileHandle,"\tRecording of %f seconds has been DISCARTED!\n", (rawData.time(end) - rawData.time(1)));
        continue;
    end
    
    MissingDataPercent = RawEcgSignalAnalysis(rawData.fs,rawData.ekg_r);
    if GRAPHICS.SHOW_INPUT_SIGNAL == true
        if ~isempty(startDateTime)
            dateTimes = startDateTime + seconds(rawData.time);
        elseif ~isempty(rawData.startDateTime)
            dateTimes = rawData.startDateTime + seconds(rawData.time);
        else
            dateTimes = datetime.empty;
        end    
        figure(111);
        if (rawData.isDifferential == true)
            subplot(2,1,1);
            plot(dateTimes,rawData.ekg_r);
            xlabel('Date Time]');
            ylabel('Raw Differential EKG value [\muvolts]');
            subplot(2,1,2);
            plot(dateTimes,-rawData.ekg_r);
            xlabel('Date Time]');
            ylabel('Raw Differential (Inverted Polarity) EKG value [\muvolts]');
        else
            plot(dateTimes,rawData.ekg_r);
            xlabel('Date Time]');
            ylabel('Raw EKG value [\muvolts]');
        end    
        grid on;
        pause(1);
    end
    
    fprintf('\tSampling Frequency: %f Hz\n', rawData.fs);
    if ~isempty(startDateTime)
       rawData.startDateTime = startDateTime;
    end
    
    fs = rawData.fs;% Sampling rate
	nSamples = length(rawData.time);
    
    PRE_PROCESSING.NB_EPOCH_SAMPLES = round(PRE_PROCESSING.EPOCH_DURATION*fs);
    PRE_PROCESSING.NB_COMPLETE_EPOCH = ceil(nSamples/PRE_PROCESSING.NB_EPOCH_SAMPLES);
    PEAK_DETECTION.NB_REFACTORY_SAMPLES = ceil(POST_PROCESSING.REFACTORY_PERIOD*fs);
  
    %% ********* Processing of each epoch segment for ORIGINAL Polarity *********
    fprintf('\tPreprocessing of ECG data and QRS detection for ORIGINAL Polarity ...\n');
    qrsCandidates = {};
    j = 1;
    startIndex = 1;
    endIndex = PRE_PROCESSING.NB_EPOCH_SAMPLES;
    epochId = 1;
    while(epochId <= PRE_PROCESSING.NB_COMPLETE_EPOCH)   
        %********* Prepare epoch segment to process **********
        %fprintf('\t\tEpoch %d of %d\n',epochId,PRE_PROCESSING.NB_COMPLETE_EPOCH);
        processingInterval = startIndex:endIndex;
        time = rawData.time(processingInterval);
        ecg_r = rawData.ekg_r(processingInterval); 

        %********** Pre-processing stage **********
        qrsPreProcesing = QrsPreProcessingPT(fs, time, ecg_r);
        if (isempty(qrsPreProcesing) == true)
            %********** Update indexes for next segment **********
            startIndex = endIndex + 1;
            if (startIndex >= nSamples)
                break;
            end    
            endIndex = startIndex + PRE_PROCESSING.NB_EPOCH_SAMPLES - 1;
            if (endIndex > nSamples)
                endIndex = nSamples;
            end 
            epochId = epochId + 1;
            continue;
        end    
                
        %********** QRS Feducial Marking **********
        qrsCandidate = QrsDetectionPT(fs,qrsPreProcesing);
        if ~isempty(qrsCandidate)
            qrsCandidate.startIndex = startIndex;
            qrsCandidate.endIndex = endIndex;
            qrsCandidates{j} = qrsCandidate;
            j = j + 1;
%             if GRAPHICS.SHOW_QRS_DETECTION == 1
%                 ShowQrsDetection(qrsPreProcesing, qrsCandidate);
%             end
        end
        
        %********** Update indexes for next segment **********
        startIndex = endIndex;
        if (startIndex >= nSamples)
            break;
        end    
        endIndex   = startIndex + PRE_PROCESSING.NB_EPOCH_SAMPLES - 1;
        if (endIndex > nSamples)
            endIndex = nSamples;
        end 
        epochId = epochId + 1;
    end % while(epochId <= PRE_PROCESSING.NB_COMPLETE_EPOCH)
    
    %% ********** Processing of QRS candidates for ORIGINAL Polarity **********
    if FILES.CREATE_LOG_FILE
        fprintf(logFileHandle,'\tORIGINAL Polarity!\n');
    end    
    if (isempty(qrsCandidates))
        if FILES.CREATE_LOG_FILE
            fprintf(logFileHandle,'\tNo QRS detected!\n');
        end
        nbQrsCandidates_OP = 0;
        rrTimeSeries_OP = double.empty;
        ibiStd_OP = NaN;
    else    
        mergedQrsCandidates = QrsCandidatesMerge(qrsCandidates, fs, rawData.timeOffset);
        nbQrsCandidates_OP = length(mergedQrsCandidates.rrIntervals);
        clear qrsCandidates; % Not needed anymore so free up some memory.
        
        %%********** Post-Processing of RR-Intervals Time Series **********    
        rrTimeSeries_OP = RRsPostProcessing(mergedQrsCandidates);
        CorrectedPercent_OP = rrTimeSeries_OP.CorrectedPercentage; % In percent
        ibiStd_OP = std(rrTimeSeries_OP.rrIntervals);
        clear mergedQrsCandidates; % Not needed anymore so free up some memory.
    end
    
    %% ********* Processing of each epoch segment for INVERTED Polarity *********
    fprintf('\tPreprocessing of EGC data and QRS detection for INVERTED Polarity ...\n');
    qrsCandidates = {};
    j = 1;
    startIndex = 1;
    endIndex = PRE_PROCESSING.NB_EPOCH_SAMPLES;
    epochId = 1;
    while(epochId <= PRE_PROCESSING.NB_COMPLETE_EPOCH)   
        %********* Prepare epoch segment to process **********
        %fprintf('\t\tEpoch %d of %d\n',epochId,PRE_PROCESSING.NB_COMPLETE_EPOCH);
        processingInterval = startIndex:endIndex;
        time = rawData.time(processingInterval);
        ecg_r = -rawData.ekg_r(processingInterval); 

        %********** Pre-processing stage **********
        qrsPreProcesing = QrsPreProcessingPT(fs, time, ecg_r);
        if (isempty(qrsPreProcesing) == true)
            %********** Update indexes for next segment **********
            startIndex = endIndex + 1;
            if (startIndex >= nSamples)
                break;
            end    
            endIndex = startIndex + PRE_PROCESSING.NB_EPOCH_SAMPLES - 1;
            if (endIndex > nSamples)
                endIndex = nSamples;
            end 
            epochId = epochId + 1;
            continue;
        end    
                
        %********** QRS Feducial Marking **********
        qrsCandidate = QrsDetectionPT(fs,qrsPreProcesing);
        if ~isempty(qrsCandidate)
            qrsCandidate.startIndex = startIndex;
            qrsCandidate.endIndex = endIndex;
            qrsCandidates{j} = qrsCandidate;
            j = j + 1;
%             if GRAPHICS.SHOW_QRS_DETECTION == 1
%                 ShowQrsDetection(qrsPreProcesing, qrsCandidate);
%             end
        end
        
        %********** Update indexes for next segment **********
        startIndex = endIndex;
        if (startIndex >= nSamples)
            break;
        end    
        endIndex   = startIndex + PRE_PROCESSING.NB_EPOCH_SAMPLES - 1;
        if (endIndex > nSamples)
            endIndex = nSamples;
        end 
        epochId = epochId + 1;
    end % while(epochId <= PRE_PROCESSING.NB_COMPLETE_EPOCH)   
   
    %% ********** Processing of QRS candidates for INVERTED Polarity **********
    if FILES.CREATE_LOG_FILE
        fprintf(logFileHandle,'\tINVERTED Polarity!\n');
    end    
    if (isempty(qrsCandidates))
        if FILES.CREATE_LOG_FILE
            fprintf(logFileHandle,'\tNo QRS detected!\n');
        end    
        nbQrsCandidates_IP = 0;
        rrTimeSeries_IP = double.empty;
        ibiStd_IP = NaN;
    else    
        mergedQrsCandidates = QrsCandidatesMerge(qrsCandidates, fs, rawData.timeOffset);
        nbQrsCandidates_IP = length(mergedQrsCandidates.rrIntervals); 
        clear qrsCandidates; % Not needed anymore so free up some memory.
  
        %%********** Post-Processing of RR-Intervals Time Series ********** 
        rrTimeSeries_IP = RRsPostProcessing(mergedQrsCandidates);
        CorrectedPercent_IP = rrTimeSeries_IP.CorrectedPercentage; % In percent
        ibiStd_IP = std(rrTimeSeries_IP.rrIntervals);
        clear mergedQrsCandidates; % Not needed anymore so free up some memory.
    end
    
    %% Decision in keeping results from either the ORIGAL or INVERTED polarity
    if ((nbQrsCandidates_OP == 0 && nbQrsCandidates_IP == 0) || ...
        isnan(ibiStd_OP) && isnan(ibiStd_IP))
        fprintf(2,"\tThe current file cannot be further processed!\n");
        continue;
    end
    
    selectedPolarity = 'ORIGINAL'; % Default: use ORIGINAL Polarity
    rrTimeSeries = rrTimeSeries_OP; 
    nbQrsCandidates = nbQrsCandidates_OP;
    CorrectedPercent = CorrectedPercent_OP;
    if (~isnan(ibiStd_OP))
        if (~isnan(ibiStd_IP))
            if (ibiStd_IP < ibiStd_OP)
                rrTimeSeries = rrTimeSeries_IP;
                nbQrsCandidates = nbQrsCandidates_IP;
                selectedPolarity = 'INVERTED';
                CorrectedPercent = CorrectedPercent_IP;
            end    
        end    
    else % ibiStd_OP == NaN
        if (~isnan(ibiStd_IP))
            rrTimeSeries = rrTimeSeries_IP;  
            nbQrsCandidates = nbQrsCandidates_IP;
            selectedPolarity = 'INVERTED';
            CorrectedPercent = CorrectedPercent_IP;
        end
    end
        
     %% Format Datetime and time into arrays
     Res.MissingPercent = MissingDataPercent;
     notMissing = -MissingDataPercent/100.0 + 1.0;
     Res.CorrectedPercent = CorrectedPercent;
     notCorrected = -CorrectedPercent/100.0 + 1.0;
     dataQualityFactor = notMissing*notCorrected;
     Res.DataQualityFactor = 100.0*dataQualityFactor;
     
     Res.StartDateTime = rawData.startDateTime;
     Res.DateTimes = Res.StartDateTime + seconds(rrTimeSeries.qrsTimeStamps);
     Res.RTimes = datestr(Res.DateTimes,'dd-mmm-yyyy HH:MM:SS.FFF');
     Res.TimeFromStart = rrTimeSeries.qrsTimeStamps';
     Res.RRs = rrTimeSeries.rrIntervals';                                           %% final cleaned output for RRI %%
     dummyStrings = char(split(allEdfFiles(i).name,'.'));
     Res.ParticipantID = dummyStrings(1,:);
    
    fprintf("\t%s polarity selected\n",selectedPolarity); 
    if FILES.CREATE_LOG_FILE
        fprintf(logFileHandle,"%s polarity selected\n",selectedPolarity);
    end    
    if (isempty(rrTimeSeries))
        fprintf(2,'\tVERY NOISY DATA FILE ... FILE NOT FULLY PROCESSED!\n');
        continue;
    else
        fprintf('\tIBI Time Series Average: %f second\n',mean(rrTimeSeries.rrIntervals));
        fprintf('\tIBI Time Series Median: %f second\n',median(rrTimeSeries.rrIntervals));
        fprintf('\tIBI Time Series STD: %f second\n', std(rrTimeSeries.rrIntervals));
        fprintf('\tMissing: %f percent\n',Res.MissingPercent);
        fprintf('\tCorrected: %f percent\n',Res.CorrectedPercent);
        fprintf('\tData Quality Factor: %f percent\n',Res.DataQualityFactor);
        if FILES.CREATE_LOG_FILE
            fprintf(logFileHandle,'Corrected IBI Average: %f second\n',mean(rrTimeSeries.rrIntervals));
            fprintf(logFileHandle,'Corrected IBI Median: %f second\n',median(rrTimeSeries.rrIntervals));
            fprintf(logFileHandle,'Corrected IBI STD: %f second\n', std(rrTimeSeries.rrIntervals));
            fprintf(logFileHandle,'Missing: %f percent\n',Res.MissingPercent);
            fprintf(logFileHandle,'Corrected: %f\n',Res.CorrectedPercent);
            fprintf(logFileHandle,'Data Quality Factor: %f percent\n\n',Res.DataQualityFactor);
        end        
    end
    
    if GRAPHICS.SHOW_RR_TIMESERIES
        currentFileName = allEdfFiles(i).name;
        idStrings = split(currentFileName,'.');
        prefix = char(idStrings(1));
        idStrings = split(prefix,'_');
        rawGraphTitle = strcat(char(join(idStrings,'-')),' - Raw RR Intervals');
        rrGraphTitle = strcat(char(join(idStrings,'-')),' - RR Intervals');
       
        figure(444);
        subplot(2,1,1)
        plot(rrTimeSeries_OP.qrsTimeStamps(2:end)./3600,rrTimeSeries_OP.rrIntervals);
        xlabel('Hours');
        ylabel('IBI (seconds)');
        xlim([0 maxTimeFromStart/3600.0]);
        title(strcat(rrGraphTitle,' - Original Polarity'));
        grid on;
        pause(1.0);
        subplot(2,1,2)
        plot(rrTimeSeries_IP.qrsTimeStamps(2:end)./3600,rrTimeSeries_IP.rrIntervals);
        xlabel('Hours');
        ylabel('IBI (seconds)');
        xlim([0 maxTimeFromStart/3600.0]);
        title(strcat(rrGraphTitle,' - Inverted Polarity'));
        grid on;
        pause(1.0);
       
        if GRAPHICS.SAVE_RR_TIMESERIES_GRAPHICS == true
            outputPngFolder = strcat(folderPath, ...
                               FILES.FILE_SEP, ...
                               'IBIs', ...
                               FILES.FILE_SEP, ...
                               'GRAPHICS');
            
            if ~exist(outputPngFolder, 'dir')
                mkdir(outputPngFolder)
            end               
                           
            outputPngFile = strcat(outputPngFolder, ...
                               FILES.FILE_SEP, ...
                               Res.ParticipantID, ...
                               FILES.PNG_TYPE);
            saveas(gcf,outputPngFile);
        end          
    end   
    
    if FILES.CREATE_IBI_TIMESERIES_CSV == true
        fprintf('Writing results to CSV file...\n');
        csvDestinationFolder = strcat(folderPath, ...
                               FILES.FILE_SEP, ...
                               'IBIs');
        if ~exist(csvDestinationFolder, 'dir')
            mkdir(csvDestinationFolder)
        end
        outputCsvFile = strcat(csvDestinationFolder, ...
                               FILES.FILE_SEP, ...
                               Res.ParticipantID, ...
                               FILES.CSV_TYPE);
        CreateIbiTimeSeriesCSV(outputCsvFile,Res);
        
    end % End of  if FILES.CREATE_IBI_TIMESERIES_CSV == true
    clear rawData;
    
end % End of for i=1:nbFiles

if FILES.CREATE_LOG_FILE
    fclose(logFileHandle);
end

end % End of EcgPipeline function
