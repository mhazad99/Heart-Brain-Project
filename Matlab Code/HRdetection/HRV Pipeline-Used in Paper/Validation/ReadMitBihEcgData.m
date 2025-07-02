function [signalData] = ReadMitBihEcgData(pathToRecords,recordName)
%READMITBIHECGDATA 

signalData.startDateTime = datetime.empty;
signalData.startDate = string.empty;
signalData.startTime = string.empty;

currentDir = pwd;
cd(pathToRecords);
recordLocation = strcat('./',recordName); 

[sig, signalData.fs, signalData.time] = rdsamp(recordLocation);

nbChannels = size(sig,2);
if (nbChannels == 2) % Two channels -> Take differential
    signalData.ekg_r = sig(:,1) - sig(:,2);
else 
    signalData.ekg_r = sig(:,1)
end

cd(currentDir);

end % End of ReadMitBihEcgData function

