function p = f_GetPath(inDir,keepDir)
%F_GETPATH - Convert a directory structure returned by dir() to cell array of full path,
% ignoring '.' and '..'.
% Do not return directory paths unless the 2nd argument is set to 1 (true).
% 
% SYNOPSIS: p = f_GetPath(inDir,keepDir)

p = '';
if nargin==1
    keepDir = false;
end
if ischar(inDir)
    sDir = dir(inDir);
else
    sDir = inDir;
end

if length(sDir) > 1
    sDir = sDir(~(strcmp({sDir.name}, '.') | strcmp({sDir.name}, '..')));
    if ~keepDir
        sDir = sDir(~[sDir(:).isdir]);
    end
    p = arrayfun(@(a) fullfile(a.folder,a.name),sDir,'UniformOutput',false);
elseif length(sDir) == 1
    if strcmp(sDir.name,'.') || strcmp(sDir.name,'..') || (~keepDir && sDir.isdir)
        p = '';
    else
        p = fullfile(sDir.folder,sDir.name);
    end
else
    warning('backtrace','off')
    warning('Directory not found or empty: %s.',inDir)
    warning('backtrace','on')
%     fprintf('WARNING: Directory not found or empty.\n')
end

end
