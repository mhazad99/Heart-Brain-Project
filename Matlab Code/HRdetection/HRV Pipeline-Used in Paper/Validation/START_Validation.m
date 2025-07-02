% START_Validation
%
% MIT-BIH arrhythmia database
currentFolder = "R:\IMHR\Sleep Research\Daniel\HRV Pipeline Validation Data\mit-bih-arrhythmia-database-1.0.0\data";

allEntries = dir(currentFolder);
nbEntries = length(allRecordFiles);
nbRecords = 0;
recordIds = string.empty;
for i=1:nbEntries
    if ~allRecordFiles(i).isdir
        nbRecords = nbRecords + 1;
        temp = split(allRecordFiles(i).name,'.');
        recordIds(nbRecords) = string(temp(1));
    end
end

cdrawData = ReadMitBihEcgData(currentFolder,char(recordIds(1)));