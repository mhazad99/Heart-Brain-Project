function WriteEkgInCsvFile(CompleteFileName,startDateTime,fs,allData)
%WriteEkgInCsvFile 

fid = fopen(CompleteFileName,'w');
fprintf(fid,'%s,%s\n',num2str(startDateTime),num2str(fs));

fprintf(fid,'%f,%f\n',allData);

fclose(fid);

end % End of WriteEkgInCsvFile
 
