function CreateCSVFileTitles(CSVFileName)
%CreateCSVFileTitles 
titles = 'Input File,Patient Name,Project,Hospital #,Subject Code,Study Date,SSC/SIN,Sex,D.O.B,Age,Height,Weight,B.M.I,';

[csvFid,msg] = fopen(CSVFileName, 'wt');
disp(msg);
fprintf(csvFid,'%s\n',titles);
fclose(csvFid);

end

