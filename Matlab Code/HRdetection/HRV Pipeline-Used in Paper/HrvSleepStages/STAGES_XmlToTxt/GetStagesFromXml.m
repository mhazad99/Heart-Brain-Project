function [stages] = GetStagesFromXml(pXmlFile)
%GETSTAGESFROMXML 

stages = int32.empty;

try
    fid = fopen(pXmlFile);
catch ex
     error('Failed to OPEN XML file %s (%s).',pXmlFile, ex.message);
end    

try
    allText = fscanf(fid,'%s');
catch ex
	fclose(fid);
	error('Failed to READ XML file %s (%s).',pXmlFile, ex.message);
end  
fclose(fid);

% First get all <SleeepStags>Stage Value</SleepSage> tags.
expression = "<SleepStage>[\d\s]</SleepStage>";
matchedData = regexp(allText,expression,'match');

nbEpochs = length(matchedData);
% Get the Sleep Stage Value
expression = "[\d\s]";

for i=1:nbEpochs
    matchedValue = regexp(string(matchedData(i)),expression,'match');
    if matchedValue == ""
        stages(i) = 9;
    else
        stages(i) = str2num(matchedValue);
    end
end

