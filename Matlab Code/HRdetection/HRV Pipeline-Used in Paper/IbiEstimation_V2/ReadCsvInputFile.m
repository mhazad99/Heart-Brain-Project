function rawData = ReadCsvInputFile(fileHandle)
%READCSCINPUTFIL
%
% The file must have succesfully be opened using fope and teh corresponding
% file handle passed as the first parameters.

global PRE_PROCESSING
formatSpecData = '%{dd/MM/yyyy HH:mm:ss.SSS}D %d';       

% If beginenning of file
if ftell(fileHandle) == 0
    formatSpecHeader = '%s %s';
	headerData = textscan(fileHandle,formatSpecHeader,1,'Delimiter',',');
    data = textscan(fileHandle,formatSpecData,PRE_PROCESSING.MAX_DATA_SEGMENT,'Delimiter',',');
    data{:,1}.TimeZone = 'America/New_York';
    rawData.time = posixtime(data{:,1})';
    PRE_PROCESSING.DATA_START_DATE_TIME = rawData.time(1);
    rawData.time = rawData.time - PRE_PROCESSING.DATA_START_DATE_TIME;
else
    data = textscan(fileHandle,formatSpecData,PRE_PROCESSING.MAX_DATA_SEGMENT,'Delimiter',',');
    data{:,1}.TimeZone = 'America/New_York';
    rawData.time = posixtime(data{:,1})';
    rawData.time = rawData.time - PRE_PROCESSING.DATA_START_DATE_TIME;
end    
rawData.ekg_r = data{:,2}';
rawData.fs = round(1/( rawData.time(2) -  rawData.time(1)));
clear data;

end % End of ReadCsvInputFile

