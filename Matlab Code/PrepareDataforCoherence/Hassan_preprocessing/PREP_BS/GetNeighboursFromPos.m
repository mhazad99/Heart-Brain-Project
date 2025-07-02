function neighbours = GetNeighboursFromPos(chanpos,n_neighbors,label)
%GETNEIGHBOURSFROMPOS - Compute neighbourhood based on euclidean distance
%
% SYNOPSIS: neighbours = GetNeighboursFromPos(chanpos,maxdist,label)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS: Got from fieldtrip ft_prepare_neighbours.m and modified to fit data
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
% Created on: 15-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nchan = length(chanpos);

% compute the distance between all sensors
dist = zeros(nchan,nchan);
for i=1:nchan
  dist(i,:) = sqrt(sum(([chanpos{:}] - repmat(chanpos{i}, 1, nchan)).^2,1));
end

%set diagonal to inf so point is not its own closest neighbor
dist(1:size(dist,1)+1:end) = inf;
%find index of closest n_neighbors
for i=1:n_neighbors
    [~, minidx(:,i)] = min(dist, [], 2);
    for j=1:length(dist)
        dist(j,minidx(j,i)) = inf;
    end
end

% % find the neighbouring electrodes based on distance
% % later we have to restrict the neighbouring electrodes to those actually selected in the dataset
% channeighbstructmat = (dist<maxdist);
% 
% % electrode istelf is not a neighbour
% channeighbstructmat = (channeighbstructmat .* ~eye(nchan));
% 
% % convert back to logical
% channeighbstructmat = logical(channeighbstructmat);

% construct a structured cell-array with all neighbours
neighbours=struct;
for i=1:nchan
    if exist('label','var')
        neighbours(i).label       = label{i};
        neighbours(i).neighblabel = label(minidx(i,:));
%         neighbours(i).neighblabel = label(channeighbstructmat(i,:));
    end
    neighbours(i).neighbidx = minidx(i,:);
%     neighbours(i).neighbidx = channeighbstructmat(i,:);
end