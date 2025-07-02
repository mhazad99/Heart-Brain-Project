function UpdateCSVInfoFile(header, CSVFileName)
%UpdateCSVInfoFile 

csvDataString = sprintf('%s,',deblank(header(1,:)));
nbElements = min(size(header));

for i=2:nbElements
   tabString = split(header(i,:),':');
   if length(tabString) == 2
      elementString = sprintf('%s,',strtrim(char(tabString(2))));
      csvDataString = strcat(csvDataString,elementString);
   else
      csvDataString = strcat(csvDataString,',');
   end   
end % End of UpdateCSVInfoFile function

csvFid = fopen(CSVFileName, 'at');
   fprintf(csvFid,'%s\n',csvDataString);
fclose(csvFid);

