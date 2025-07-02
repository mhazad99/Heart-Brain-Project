function fid = WriteVMRKHeader(fullFileName,recordingFileName,recStartTime)
%WRITEVMRKHEADER - Write a VMRK marker file header and recording start date 
%   and time as first marker.
%
% SYNOPSIS: WriteVMRKHeader(fullFileName,recordingFileName,recStartTime)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 
%
% Copyright Tomy Aumont

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1135713 (R2019a) Update 3 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 11-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid = fopen(fullFileName,'w');

if fid == -1
    error('ERROR: Could not write file %s\n',fullFileName);
end

fprintf(fid,'Brain Vision Data Exchange Marker File, Version 1.0\n');
fprintf(fid,'\n');
fprintf(fid,'[Common Infos]\n');
fprintf(fid,'Codepage=UTF-8\n');
fprintf(fid,'DataFile=%s\n',recordingFileName);
fprintf(fid,'\n');
fprintf(fid,'[Marker Infos]\n');
fprintf(fid,'; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n');
fprintf(fid,'; <Size in data points>, <Channel number (0 = marker is related to all channels)>\n');
fprintf(fid,'; Fields are delimited by commas, some fields might be omitted (empty).\n');
fprintf(fid,'; Commas in type or description text are coded as "\\1".\n');
fprintf(fid,'Mk1=New Segment,,1,1,0,%s\n',recStartTime);

fclose(fid);










    