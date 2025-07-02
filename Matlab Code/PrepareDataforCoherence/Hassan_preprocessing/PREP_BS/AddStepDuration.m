function arr = AddStepDuration(arr,stepTag, stepTime, isPrint)
% SYNOPSIS: AddStepDuration()
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
% Created on: 30-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Append arr with a new entry and update total time
 arr = [arr; {stepTag,stepTime}];
 arr{1,2} = sum([arr{2:end,2}]);
 if nargin < 4 || isPrint == 1
     fprintf('PREP>\t%s done in %f seconds\n',arr{end,1},arr{end,2})
 end
