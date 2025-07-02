function sleepStages = ReadSleepStages(stagingFileName)
%ReadSleepStages 
global EPOCH;

    headerCell = {};
    datacell = {};
    fid = fopen(stagingFileName,'r');
    
    idStrings = split(stagingFileName,'\');
    
    %********** File Header **********
    i = 1;
    isHeader = true;
    while(isHeader == true)
        headerCell = string(fgetl(fid));
        
        if contains(headerCell, 'Event','IgnoreCase',true) || ...
           contains(headerCell, 'Epoch','IgnoreCase',true)
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
        epochNumber = str2num(dataCellStrings(1));
        %fprintf("\tEpoch #%d\n",epochNumber);
        
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
                            sleepStages.encoding(i) = 1;
                    end % End of switch
                else
                    fprintf(2,'\tNo number associated to NREM stage\n');
                    sleepStages.stageType(i) = 'N1';
                    sleepStages.encoding(i) = 1;
                end
            end
        else
            sleepStages.stageType(i) = 'W';
            sleepStages.encoding(i) = 0;
        end
        sleepStages.stageTime(i) = (i-1)*EPOCH.DURATION;
            
        previousEpochId = sleepStages.epochs(i);
        i = i+1;
    end  
    
    fclose(fid);
       
end % End of ReadSleepStages function

