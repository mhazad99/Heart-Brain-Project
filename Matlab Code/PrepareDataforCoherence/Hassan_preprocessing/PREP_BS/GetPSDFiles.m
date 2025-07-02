function fileList = GetPSDFiles(baseDir,subjects)
%GETPSDFILES - Get absolute path of PSD files of each subject given
% SYNOPSIS: GetPSDFiles()
%
%       baseDir:    Protocol directory containing subjects
%       subjects:   cell array of subject names
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
% Created on: 24-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start brainstorm if not already running
if ~brainstorm('status')
    brainstorm nogui
end

sStudies = bst_get('ProtocolStudies');
sTimeFreq = [sStudies.Study.Timefreq];
fileList = {};
for iSubj = 1:length(subjects)
    iFiles = contains({sTimeFreq.FileName},subjects{iSubj},'IgnoreCase',true) & ...
        ~contains({sTimeFreq.FileName},'@raw','IgnoreCase',true);
%      fileList(:,iSubj) = cellfun(@(c) bst_fullfile(baseDir,c),sort({sTimeFreq(iFiles).FileName})','UniformOutput',false);
     fl = sort({sTimeFreq(iFiles).FileName})';
     cDate = split(fl,{'psd_','_'});
     uniqueDates = unique(cDate(:,end-1));
     if size(uniqueDates,1) > 1
         menuList = datestr(datetime(uniqueDates,'InputFormat','yyMMdd'));
         [iS,~] = listdlg('ListString',menuList,'SelectionMode','single','PromptString',['Select one PSD date: ' subjects{iSubj}]);
         if ~iS; continue; end
         fileList(:,iSubj) = fl(contains(fl,uniqueDates{iS}));
     else
         % Keep latest from single date list of files
         [~,slp_stage_list] = cellfun(@fileparts,cellfun(@fileparts,fl,'UniformOutput',false),'UniformOutput',false);
         nbr_psd_runs       = max(count(slp_stage_list,sleep_stages));
         fileList(:,iSubj)  = fl(nbr_psd_runs:nbr_psd_runs:end);
%          fileList(:,iSubj) = fl;
     end
end
% dirs = cellfun(@(s) fullfile(baseDir,'data',char(s),'@intra'),subjects,'UniformOutput',false);
% nFilesMax = 0;
% absPath = cell(length(dirs),1);
% for iDir = 1:length(dirs)
%     newFiles = dir([dirs{iDir} filesep '*_psd_*']);
%     if ~isempty(newFiles)
%         absPath{iDir} = fullfile({newFiles.folder},{newFiles.name})';
%         if length(absPath{iDir}) > nFilesMax
%             nFilesMax = length(absPath{iDir});
%         end
%     end
% end
% fileList = cell(length(dirs),nFilesMax);
% for iDir = 1:length(dirs)
%     fileList(:,iDir) = [absPath{iDir}; cell(nFilesMax-length(absPath{iDir}),1)];
% end


