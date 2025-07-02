function scoringFile = FindScoringFileFromId(folder, id)
%FindScoringFileFromId 

    scoringFile = string.empty;
    allFiles = dir(strcat(folder,filesep,"*.txt"));
    nbFiles = length(allFiles);

    for i=1:nbFiles
        if contains(allFiles(i).name,id,'IgnoreCase',true) &&  ...
           (~contains(allFiles(i).name,"tag",'IgnoreCase',true) || ...
            contains(allFiles(i).name,"stage",'IgnoreCase',true))

            scoringFile = allFiles(i).name;
            break;
        end
    end

end % FindScoringFileFromId

