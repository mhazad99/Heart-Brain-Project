function [fPath,fName,fExt] = BackupFileWithNumber(fSrc)
% SYNOPSIS: BackupFileWithNumber
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.5.0.1049112 (R2018b) Update 3 on
%    Linux 5.0.10-arch1-1-ARCH #1 SMP PREEMPT Sat Apr 27 20:06:45 UTC 2019 x86_64
%
% Author:     Tomy Aumont
% Work:       University of Montreal
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 09-May-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[fPath, srcfName, fExt] = fileparts(fSrc);
fName = [srcfName,'_orig'];
destFileName = fullfile(fPath, [fName,fExt]);

if isempty(fExt)  % No '.mat' in FileName
  fExt     = '.mat';
  destFileName = fullfile(fPath, [fName, fExt]);
end

% Verify to create only one original and then backups
if exist(destFileName,'file') == 2 % R2017b or later -> isfile(destFileName)
    fName = [srcfName,'_bckup'];
    destFileName = fullfile(fPath, [fName,fExt]);
end

while exist(destFileName,'file') == 2 % R2017b or later -> isfile(destFileName)
  % Get number of files:
  n = strfind(fName,'bckup') + 5;
  fNum = sscanf(fName(n:end), '%d');
  if isempty(fNum)
      fName = [fName '1'];
  else
      fName = strrep(fName,num2str(fNum),num2str(fNum+1)); % update number
  end
  destFileName = fullfile(fPath,[fName,fExt]);
end

fprintf('RL2BV>     Creating backup file: %s\n',destFileName);
copyfile(fSrc, destFileName);
