function rawData = ReadEcgInputFile(inputFileName)
%READECGINPUTFILE
rawData.startDateTime = datetime.empty;
rawData.startDate = string.empty;
rawData.startTime = string.empty;

[header, recordData] = edfread(inputFileName);

% Find the EKG channels
[~,ekgChan] = find(contains(split(header.label),'ekg','IgnoreCase',true)); %% the label for ekg should be changed!
if ~isempty(ekgChan)
    if (length(ekgChan) == 1) % Single channel
        rawData.ekg_r = recordData(ekgChan,:);
        rawData.fs = header.frequency(ekgChan); %% if the recorded file does not contain the frequency header you should add the
        %frequency!
        rawData.isDifferential = false;
        fprintf('\tSINGLE EKG channel\n');
    else % Differential recording    
        rawData.ekg_r = (recordData(ekgChan(1),:) - recordData(ekgChan(2),:));
        rawData.fs = header.frequency(ekgChan(1));
        rawData.isDifferential = true;
        fprintf('\tDIFFERENTIAL EKG channel\n');
    end
else    
    [~,ekgChan] = find(contains(split(header.label),'ECG','IgnoreCase',true));
    if isempty(ekgChan)
        rawData.ekg_r = recordData(1,:);
        rawData.fs = header.frequency(1);
        rawData.isDifferential = false;
    elseif (length(ekgChan) == 1) % Single channel
        rawData.ekg_r = recordData(ekgChan,:);
        rawData.fs = header.frequency(ekgChan);
        rawData.isDifferential = false;
        fprintf('\tSINGLE EKG channel\n');
    else % Differential recording    
        rawData.ekg_r = (recordData(ekgChan(2),:) - recordData(ekgChan(1),:));
        rawData.fs = header.frequency(ekgChan(1));
        rawData.isDifferential = true;
        fprintf('\tDIFFERENTIAL EKG channel\n');
    end
end   

% Generating time from start using the Sampling Frequency
rawData.time = (0:length(rawData.ekg_r)-1)/rawData.fs; % Time in seconds

if ~isempty(header.startdate) && ~isempty(header.starttime)
    % Create a datetime object corresponding the current date since only the time is important.
    %dummyStrs = char(split(header.startdate,'.'));
    currentDateTime = datetime();
%     dd = str2num(dummyStrs(1,:));
%     mm = str2num(dummyStrs(2,:));
%     yy = str2num(dummyStrs(3,:));
%     yyyy = yy + 2000;
    dd = currentDateTime.Day;
	mm = currentDateTime.Month;
	yyyy = currentDateTime.Year;

    dummyStrs = char(split(header.starttime,'.'));
	hh = str2num(dummyStrs(1,:));
    MI = str2num(dummyStrs(2,:));
    ss = str2num(dummyStrs(3,:));
    
    rawData.startDate = header.startdate;
    rawData.startTime = header.starttime;
    rawData.startDateTime = datetime(yyyy,mm,dd,hh,MI,ss);
end

clear header recordData ekgChan;

end % End of ReadEcgInputFile

