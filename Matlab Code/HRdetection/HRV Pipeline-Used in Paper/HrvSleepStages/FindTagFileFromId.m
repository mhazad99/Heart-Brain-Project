function tagFile = FindTagFileFromId(folder, id)
%FindTagFileFromId 

    tagFile = string.empty;
    allFiles = dir(strcat(folder,filesep,"*.txt"));
    nbFiles = length(allFiles);

    for i=1:nbFiles
        if contains(allFiles(i).name,id,'IgnoreCase',true) &&  ...
           (contains(allFiles(i).name,"tag",'IgnoreCase',true) && ...
            ~contains(allFiles(i).name,"stage",'IgnoreCase',true))

            tagFile = allFiles(i).name;
            break;
        end
    end

end % FindTagFileFromId

