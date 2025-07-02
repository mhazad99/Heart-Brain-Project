function sleepStages = ReadSleepStagesFormat1(stagingFileName)
%READSLEEPSTAGESFORMAT1 

    SECONDS_PER_EPOCH = 30.0;
    headerCell = {};
    datacell = {};
    fid = fopen(stagingFileName,'r');
    
    idStrings = split(stagingFileName,'\');
    fprintf('Reading and processing %s file ...\n', string(idStrings(end)));
    
    %********** File Header **********
    i = 1;
    isHeader = true;
    while(isHeader == true)
        headerCell = fgetl(fid);
        if contains(headerCell, 'Patient:','IgnoreCase',true) 
            dummyStrs = split(char(headerCell),':');
            sleepStages.PatientId = strtrim(char(dummyStrs{2}));
        end 
        
        if contains(headerCell, 'Recording Date:','IgnoreCase',true) 
            dummyStrs = split(char(headerCell),':');
            % Format: dd/mm/yyyy
            sleepStages.RecordingDateString = strtrim(char(dummyStrs{2}));
            dummyStrs = split(sleepStages.RecordingDateString,'/');
            d = str2num(char(dummyStrs(1)));
            m = str2num(char(dummyStrs(2)));
            y = str2num(char(dummyStrs(3)));
         end  
        
        if contains(headerCell, 'Time','IgnoreCase',true) 
            isHeader = false;
        end  
           
        i = i + 1;
    end    
    
    %********** Data **********
    sleepStages.stageType = string.empty;
    sleepStages.stageTime = string.empty;
	sleepStages.stageDateTime = datetime.empty;
    epochId = 1;   
    while (~feof(fid))
        dataCell = fgetl(fid);
        dataCellStrings = split(dataCell); % space delimiter
        if (length(dataCellStrings) < 4)
            continue;
        end
        sleepStages.stageTime(epochId) = char(dataCellStrings(1)); % hh:mm:ss
        AmPm = char(dataCellStrings(2)); % AM or PM
        sleepStages.stageType(epochId) = char(dataCellStrings(3));
        
        % Create a datetime object for the current stage
        dummyStrs = split(sleepStages.stageTime(epochId),':');
        h = str2num(char(dummyStrs(1)));
        mnt = str2num(char(dummyStrs(2)));
        s = str2num(char(dummyStrs(3)));
        
        if contains(AmPm,'PM')
            if h<12
                h = h + 12;                
            end
        elseif contains(AmPm,'AM') && h==12
            h = 0;
        end   
        
        if epochId == 1
            referenceDateTime = datetime(y,m,d,h,mnt,s);
            sleepStages.stageDateTime(epochId) = referenceDateTime;
        else
            sleepStages.stageDateTime(epochId) = referenceDateTime + ...
                seconds((epochId-1)*SECONDS_PER_EPOCH);
        end
      
        epochId = epochId + 1;
    end    
    
    fclose(fid);
    
    nbEpochs = length(sleepStages.stageDateTime);
    % All wake stages : 'W'
    sleepStages.idxWake = ...
        find(contains(sleepStages.stageType,'W','IgnoreCase',true));
    if (~isempty(sleepStages.idxWake))
        sleepStages.wakeStartTimes = sleepStages.stageDateTime(sleepStages.idxWake);
        if (sleepStages.idxWake(end) == nbEpochs)
            sleepStages.wakeEndTimes = ...
                [sleepStages.stageDateTime(sleepStages.idxWake(2:end)) ...
                 sleepStages.stageDateTime(sleepStages.idxWake(end)) + minutes(30)];
        else
            sleepStages.wakeEndTimes = sleepStages.stageDateTime(sleepStages.idxWake+1);
        end       
    else
        sleepStages.wakeStartTimes = [];
        sleepStages.wakeEndTimes = [];
    end
    
     % All REM stages: 'R'
    sleepStages.idxRem = ...
        find(contains(sleepStages.stageType,'R','IgnoreCase',true) == true);
    if (~isempty(sleepStages.idxRem))    
        
        sleepStages.remStartTimes = sleepStages.stageDateTime(sleepStages.idxRem);
        if (sleepStages.idxRem(end) == nbEpochs)
            sleepStages.remEndTimes = ...
                [sleepStages.stageDateTime(sleepStages.idxRem(2:end)) ...
                 sleepStages.stageDateTime(sleepStages.idxRem(end)) + minutes(30)];
        else
            sleepStages.remEndTimes = sleepStages.stageDateTime(sleepStages.idxRem+1);
        end 
        
    else
        sleepStages.remStartTimes = [];
        sleepStages.remEndTimes = [];
    end
    
    % All NREM1 stages : N1
	sleepStages.idxNrem1 = ...
        find(contains(sleepStages.stageType,'N1','IgnoreCase',true) == true);
    if (~isempty(sleepStages.idxNrem1))    
        
        sleepStages.nrem1StartTimes = sleepStages.stageDateTime(sleepStages.idxNrem1);
        if (sleepStages.idxNrem1(end) == nbEpochs)
            sleepStages.nrem1EndTimes = ...
                [sleepStages.stageDateTime(sleepStages.idxNrem1(2:end)) ...
                 sleepStages.stageDateTime(sleepStages.idxNrem1(end)) + minutes(30)];
        else
            sleepStages.nrem1EndTimes = sleepStages.stageDateTime(sleepStages.idxNrem1+1);
        end 
        
    else
        sleepStages.nrem1StartTimes = [];
        sleepStages.nrem1EndTimes = [];
    end
    
    % All NREM 2 stages : N2
    sleepStages.idxNrem2 = ...
        find(contains(sleepStages.stageType,'N2','IgnoreCase',true) == true);
    if (~isempty(sleepStages.idxNrem2))    
        
        sleepStages.nrem2StartTimes = sleepStages.stageDateTime(sleepStages.idxNrem2);
        if (sleepStages.idxNrem2(end) == nbEpochs)
            sleepStages.nrem2EndTimes = ...
                [sleepStages.stageDateTime(sleepStages.idxNrem2(2:end)) ...
                 sleepStages.stageDateTime(sleepStages.idxNrem2(end)) + minutes(30)];
        else
            sleepStages.nrem2EndTimes = sleepStages.stageDateTime(sleepStages.idxNrem2+1);
        end 
        
    else
        sleepStages.nrem2StartTimes = [];
        sleepStages.nrem2EndTimes = [];
    end    
    
   % All NREM 3 stages : N3
    sleepStages.idxNrem3 = ...
        find(contains(sleepStages.stageType,'N3','IgnoreCase',true) == true);
    if (~isempty(sleepStages.idxNrem3))    
        
        sleepStages.nrem3StartTimes = sleepStages.stageDateTime(sleepStages.idxNrem3);
        if (sleepStages.idxNrem3(end) == nbEpochs)
            sleepStages.nrem3EndTimes = ...
                [sleepStages.stageDateTime(sleepStages.idxNrem3(2:end)) ...
                 sleepStages.stageDateTime(sleepStages.idxNrem3(end)) + minutes(30)];
        else
            sleepStages.nrem3EndTimes = sleepStages.stageDateTime(sleepStages.idxNrem3+1);
        end 
        
    else
        sleepStages.nrem3StartTimes = [];
        sleepStages.nrem3EndTimes = [];
    end  


end % End of ReadSleepStagesFormat1 function

