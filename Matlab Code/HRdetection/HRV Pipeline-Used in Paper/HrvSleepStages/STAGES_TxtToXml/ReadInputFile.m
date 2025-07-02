function [header,data] = ReadInputFile(fileName)
% ReadInputFile 
nbHeaderLines = 12;
headercell = {};
datacell = {};
WAKE = 'Wake';
NREM = 'NREM';
REM  = 'REM';

fid = fopen(fileName);

% Read the file header
headercell{1} = fileName;
for i=1:nbHeaderLines
   headercell{i+1} = fgetl(fid);
end
emptyLine =  fgets(fid);
emptyLine =  fgets(fid);

if (~isempty(headercell))
   header = char(headercell);
else
   header = [];
end  


j = 1;
while ~feof(fid)
   rawData = split(fgetl(fid));
   if length(rawData) == 1
      datacell{j} = 0;   
   elseif length(rawData) == 2
      % 'Wake' or 'REM' 
      if (strcmp(char(rawData{2}),WAKE) == 1) 
          datacell{j} = 0;
      elseif  (strcmp(char(rawData{2}),REM) == 1)
          datacell{j} = 5;
      else
         fprintf(2,'ERROR inWake or REM');
      end   
   elseif length(rawData) == 3  
       if (strcmp(char(rawData{2}),NREM) == 1) 
          datacell{j} = str2num(rawData{3});
       else 
          fprintf(2,'ERROR in NREM');
       end
   else
      fprintf(2,'ERROR length(rawData)');
   end
   j = j + 1;
end   
   
if (~isempty(datacell))
   data = cell2mat(datacell);
else
   data = [];
end   

fclose(fid);

end % End of ReadInputFile function

