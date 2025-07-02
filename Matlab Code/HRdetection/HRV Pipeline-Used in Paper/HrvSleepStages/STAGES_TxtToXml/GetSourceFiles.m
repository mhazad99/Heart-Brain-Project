function [inputFileNames, baseFileNames] = GetSourceFiles(sourceFolder)
% GetSourceFiles returns a cell array of string corresponding to the data
% files to process.

validPattern = 'Patient Name';
validFileNames = {};

folder = strcat(sourceFolder,filesep,'*.txt');

files = dir(folder);

% There are some files in the directory.
nbFiles = length(files);
if (nbFiles > 0)
   j = 1;
   for i=1:nbFiles
      fileName = strcat(strcat(sourceFolder,filesep),files(i).name);
      fid = fopen(fileName);
      firstLine = split(fgetl(fid),':');
      
      % This is a valid input text file.
      if (strcmp(validPattern, string(firstLine(1,:))))
         validFileNames{j} = fileName;
         baseFileNames{j} = files(i).name;
         j = j + 1;
      end   
      fclose(fid);
   end  
end   

if exist('validFileNames','var') == 1
   inputFileNames = validFileNames;
else
   inputFileNames = [];
end

end % End of GetSourceFiles function

