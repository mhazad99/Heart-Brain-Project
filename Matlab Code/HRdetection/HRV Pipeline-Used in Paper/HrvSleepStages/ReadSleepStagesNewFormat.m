function sleepStages = ReadSleepStagesNewFormat(stagingFileName)
%ReadSleepStages 
global EPOCH;

    headerString = "";
    startDate = "";
    fid = fopen(stagingFileName,'r');
    
    sleepStages.StartDateTime = datetime.empty;
    
    %********** File Header **********
    i = 1;
    isHeader = true;
    while(isHeader == true)
        headerString = string(fgetl(fid));
        
        if contains(headerString, 'Recording Date','IgnoreCase',true) 
            dummy = strsplit(headerString,':');
            startDate = strtrim(string(dummy(2)));
        end
        
        if contains(headerString, 'Time','IgnoreCase',true) 
            isHeader = false;
        end  
    end    
    
    %********** Data **********
    sleepStages.epochs = int32.empty;
	sleepStages.stageType = string.empty;
    sleepStages.encoding = int32.empty;
    sleepStages.stageTime = double.empty;
	sleepStages.lightsOff = NaN;
    sleepStages.lightsOn  = NaN;
    sleepStages.valide = true;
    
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
        epochNumber = i;
        
        % First epoch
        if i == 1 && ~isempty(startDate)
            startDateTime = strcat(startDate," ",dataCellStrings(1)," ",dataCellStrings(2));   
            sleepStages.StartDateTime = datetime(startDateTime, ...
                                                 'InputFormat', ...
                                                 'dd/MM/yyyy hh:mm:ss a');
        end    
        
        % Lights OFF
        if contains(dataCellStrings,'lights off','IgnoreCase',true) == true
            sleepStages.lightsOff = epochNumber
        end
        
        % Lights ON
        if contains(dataCellStrings,'lights on','IgnoreCase',true) == true
            sleepStages.lightsOn = epochNumber
        end
        
        % This is to remove extra codes for the same epoch that appears in 
        % some sleep scoring files. Those lines should be ignored.
        if  epochNumber == previousEpochId
            continue;
        end    
         
        sleepStages.epochs(i) = epochNumber;
        sleepStages.stageTime(i) = (i-1)*EPOCH.DURATION;
        sleepStages.stageType(i) = strtrim(dataCellStrings(3));
        if contains(sleepStages.stageType(i),"W")
            sleepStages.encoding(i) = 0;
        elseif contains(sleepStages.stageType(i),"N1")    
            sleepStages.encoding(i) = 1;
        elseif contains(sleepStages.stageType(i),"N2")    
            sleepStages.encoding(i) = 2;
        elseif contains(sleepStages.stageType(i),"N3")    
            sleepStages.encoding(i) = 3;
        elseif contains(sleepStages.stageType(i),"R")    
            sleepStages.encoding(i) = 5;
        else
            sleepStages.stageType(i) = "W";
            sleepStages.encoding(i) = 0;
        end    
            
        previousEpochId = sleepStages.epochs(i);
        i = i+1;
    end  
    
    fclose(fid);
       
end % End of ReadSleepStages function

