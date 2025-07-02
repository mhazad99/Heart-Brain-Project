function [headerData,rawData] = ReadEcgEdfFile(inputFileName)
%READECGINPUTFILE
rawData.startDateTime = datetime.empty;
rawData.startDate = string.empty;
rawData.startTime = string.empty;

[headerData, recordData] = edfread1(inputFileName);

% Find the EKG channels
[~,ekgChan] = find(contains(split(headerData.label),'ekg','IgnoreCase',true));
if ~isempty(ekgChan)
    if (length(ekgChan) == 1) % Single channel
        rawData.ekg_r = recordData(ekgChan,:);
        rawData.fs = headerData.frequency(ekgChan);
    else % Differential recording    
        rawData.ekg_r = (recordData(ekgChan(1),:) - recordData(ekgChan(2),:));
        rawData.fs = headerData.frequency(ekgChan(1));
    end
else    
    [~,ekgChan] = find(contains(split(headerData.label),'ECG','IgnoreCase',true));
    if isempty(ekgChan)
        rawData.ekg_r = recordData(1,:);
        rawData.fs = headerData.frequency(1);
    elseif (length(ekgChan) == 1) % Single channel
        rawData.ekg_r = recordData(ekgChan,:);
        rawData.fs = headerData.frequency(ekgChan);
    else % Differential recording    
        rawData.ekg_r = (recordData(ekgChan(2),:) - recordData(ekgChan(1),:));
        rawData.fs = headerData.frequency(ekgChan(1));
    end
end   

% Generating time from start using the Sampling Frequency
rawData.time = (0:length(rawData.ekg_r)-1)/rawData.fs; % Time in seconds

if ~isempty(headerData.startdate) && ~isempty(headerData.starttime)
    % Create a datetime object corresponding to the start of the data.
    dummyStrs = char(split(headerData.startdate,'.'));
    dd = str2num(dummyStrs(1,:));
    mm = str2num(dummyStrs(2,:));
    yy = str2num(dummyStrs(3,:));
    if yy < 50
        yyyy = yy + 2000;
    else
        yyyy = yy + 1900;
    end
    
    dummyStrs = char(split(headerData.starttime,'.'));
	hh = str2num(dummyStrs(1,:));
    MI = str2num(dummyStrs(2,:));
    ss = str2num(dummyStrs(3,:));
    
    rawData.startDate = headerData.startdate;
    rawData.startTime = headerData.starttime;
    rawData.startDateTime = datetime(yyyy,mm,dd,hh,MI,ss);
end

clear header recordData ekgChan;

end % End of ReadEcgEdfFile

