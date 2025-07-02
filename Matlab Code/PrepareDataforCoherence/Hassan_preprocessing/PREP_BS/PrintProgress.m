function PrintProgress(it,maxVal,isNewLine)
% SYNOPSIS: PrintProgress()
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
% Created on: 21-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    isNewLine = false;
end

if it > 1 && ~isNewLine
    % number of digits in iteration number
    itLen = fix(abs(log10(abs(it-1))))+1;
    % number of digits in max number
    maxLen = fix(abs(log10(abs(maxVal))))+1;
    % number of character to erase = sum of number digits
    eraseLen = maxLen + itLen + 3; % 3 for "/", ")" and carriage return
    % erase them...
    fprintf(repmat('\b', 1, eraseLen));
end
% print current iteration
fprintf('(%d/%d)',it,maxVal);
if it == maxVal; fprintf('\n'); end
