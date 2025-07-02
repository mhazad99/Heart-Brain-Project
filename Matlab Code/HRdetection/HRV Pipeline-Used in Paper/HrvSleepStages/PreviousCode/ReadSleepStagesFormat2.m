function sleepStages = ReadSleepStagesFormat2(stagingFileName, inputDateTime)
%READSLEEPSTAGESFORMAT2 

    SECONDS_PER_EPOCH = 30.0;
    headerCell = {};
    datacell = {};
    fid = fopen(stagingFileName,'r');
    
    idStrings = split(stagingFileName,'\');
    
    %********** File Header **********
    i = 1;
    isHeader = true;
    while(isHeader == true)
        headerCell = fgetl(fid);
        if  contains(headerCell, 'Subject Code:','IgnoreCase',true)
            dummyStrs = split(char(headerCell),':');
            sleepStages.FormatType = 2;
            sleepStages.PatientId = strtrim(char(dummyStrs{2}));
        end 
        
        if contains(headerCell, 'Study Date:','IgnoreCase',true)                    
            dummyStrs = split(char(headerCell),':');
            if length(dummyStrs) > 3
                dummyStrs = strcat(string(strtrim(dummyStrs(2))),':',string(dummyStrs(3)),':',string(dummyStrs(4)));
                try
                    readDateTime = datetime(string(dummyStrs),'InputFormat','MM/dd/yyyy h:m:s a');
                catch ex
                    %readDateTime = datetime(string(dummyStrs),'InputFormat','M/d/yyyy h:mm:ss a');
                    readDateTime = datetime(string(dummyStrs),'InputFormat','dd/MM/yyyy h:mm:ss a');
                end    
            else
                dummyStrs = strcat(string(strtrim(dummyStrs(2))),':',string(dummyStrs(3)));
                try
                    readDateTime = datetime(string(dummyStrs),'InputFormat','MM/dd/yyyy HH:mm');
                catch ex
                    %readDateTime = datetime(string(dummyStrs),'InputFormat','M/d/yyyy HH:mm');
                    readDateTime = datetime(string(dummyStrs),'InputFormat','dd/MM/yyyy HH:mm');
                end    
            end    
%             currentDateTime = datetime();
%             dd = currentDateTime.Day;
%             mm = currentDateTime.Month;
%             yyyy = currentDateTime.Year;
            dd = inputDateTime.Day;
            mm = inputDateTime.Month;
            yyyy = inputDateTime.Year;
            hh = readDateTime.Hour;
            MI = readDateTime.Minute;
            ss = readDateTime.Second;
            sleepStages.StudyDateTime = datetime(yyyy,mm,dd,hh,MI,ss);
            sleepStages.RecordingDateString = datestr(sleepStages.StudyDateTime,'dd/MM/yyyy');
         end  
        
        if contains(headerCell, 'Event','IgnoreCase',true)   
            isHeader = false;
        end  
    end    
    
    %********** Data **********
    sleepStages.epochs = int32.empty;
	sleepStages.stageType = string.empty;
    sleepStages.encoding = int32.empty;
    sleepStages.stageTime = string.empty;

    i = 1;    
    previousEpochId = 0;
    while (~feof(fid))
        dataCell = fgetl(fid);
        dataCellStrings = string(split(dataCell)); % white space delimiter
        % The following is in case there is an emty line at the end of the
        % file.
        if isempty(char(dataCellStrings(1)))
            continue;
        end    
        epochNumber = str2num(dataCellStrings(1));
        %fprintf("\tEpoch #%d\n",epochNumber);
        
        % This is to remove extra codes for the same epoch that appears in 
        % some sleep scoring files. Those lines should be ignored.
        if  epochNumber == previousEpochId
            continue;
        end    
        
        sleepStages.epochs(i) = epochNumber;
        if (length(dataCellStrings) > 1)
            if (strcmpi(dataCellStrings(2),'Wake') || ... 
                strcmpi(dataCellStrings(2),'-') || ...
                strcmpi(dataCellStrings(2),""))
                sleepStages.encoding(i) = 0;
                sleepStages.stageType(i) = 'W';
            elseif (strcmpi(dataCellStrings(2),'REM'))    
                 sleepStages.stageType(i) = 'R';
                 sleepStages.encoding(i) = 5;
            elseif (strcmpi(dataCellStrings(2),'NREM') || strcmpi(dataCellStrings(2),'Stage'))        
                if (length(dataCellStrings) > 2)
                    nremId = str2num(dataCellStrings(3));
                    switch nremId
                        case 1
                            sleepStages.stageType(i) = 'N1';
                            sleepStages.encoding(i) = 1;
                        case 2    
                            sleepStages.stageType(i) = 'N2';
                             sleepStages.encoding(i) = 2;
                        case 3    
                            sleepStages.stageType(i) = 'N3';
                            sleepStages.encoding(i) = 3;
                        case 4    
                            sleepStages.stageType(i) = 'N3';
                            sleepStages.encoding(i) = 3;
                        otherwise
                            fprintf(2,'\tNeither N1, N2 or N3 (%d)\n',nremId);
                            sleepStages.stageType(i) = 'N1';
                            sleepStages.encoding(i) = -1;
                    end % End of switch
                else
                    fprintf(2,'\tNo number associated to NREM stage\n');
                    sleepStages.stageType(i) = 'N1';
                    sleepStages.encoding(i) = -1;
                end
            end
        else
            sleepStages.stageType(i) = 'W';
            sleepStages.encoding(i) = 0;
        end
        
        sleepStages.stageDateTime(i) = ...
                sleepStages.StudyDateTime + ...
                seconds((i-1)*SECONDS_PER_EPOCH);
        sleepStages.stageTime(i) = ...
                datestr(sleepStages.stageDateTime(i),'HH:MM:SS');
            
        previousEpochId = sleepStages.epochs(i);
        i = i+1;
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

end % End of ReadSleepStagesFormat2 function

