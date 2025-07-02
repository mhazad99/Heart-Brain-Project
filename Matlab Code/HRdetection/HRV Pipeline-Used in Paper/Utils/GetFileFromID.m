function fullPathFileName = GetFileFromID(folderLocation,fileSuffix,fileID)
%GETFILEFROMID 
% The file suffix must in the following format:
%   *.suffix
% Example: *.csv, *.txt, *.rec, *.edf, ...
%
fullPathFileName = "";

allEntries = dir(strcat(folderLocation,filesep,fileSuffix));
nbEntries = length(allEntries);

for i=1:nbEntries
    if (allEntries(i).isdir == false && ...
        contains(allEntries(i).name,fileID,'IgnoreCase',true))

        fullPathFileName = string(strcat(allEntries(i).folder,filesep,allEntries(i).name));
        break;
    end    
end    

end % End of GetFileFromID

