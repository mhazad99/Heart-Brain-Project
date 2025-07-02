function rawData = ReadEcgCsvFile(inputFileName)
%READECGINPUTFILE
rawData.startDateTime = datetime.empty;
rawData.startDate = string.empty;
rawData.startTime = string.empty;

allData = csvread(inputFileName);

rawData.fs = allData(1,2);
rawData.startDateTime = datetime(allData(1,1), 'ConvertFrom', 'posixtime');
[nbRows,nbColumns] = size(allData);
rawData.time  = allData(2:nbRows,1)'; % Seconds
rawData.ekg_r = allData(2:nbRows,2)'; % microVolts
rawData.isDifferential = true;

clear allData;

end % End of ReadEcgCsvFile

