function WriteTextFile(baseFileName, destinationFolder, stagesNum)
%WriteTextFile 

textFileName = strcat(destinationFolder,filesep,baseFileName,".txt");
try
    fid = fopen(textFileName,'w');
catch ex
     error('Failed to CREATE text file %s (%s).',textFileName, ex.message);
end    

% First write the file heaser.
fprintf(fid,GetHeader());

% Write the stages for each epoch
nbEpoch = length(stagesNum);
for i=1:nbEpoch
    switch stagesNum(i)
        case 0
            epochText = sprintf("%d Wake\n", i);
        case 1
            epochText = sprintf("%d NREM 1\n", i);
        case 2
            epochText = sprintf("%d NREM 2\n", i);
        case 3
            epochText = sprintf("%d NREM 3\n", i);
        case 4
            epochText = sprintf("%d NREM 3\n", i);
        case 5
            epochText = sprintf("%d REM\n", i);     
        otherwise
            epochText = sprintf("%d\n", i); 
    end 
    fprintf(fid,epochText);
end    

fclose(fid);

end % End of WriteTextFile

