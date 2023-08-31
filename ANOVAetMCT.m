clc
clear
close all
%% To plot the differences of coherences for sleep stages
% addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\SleepStagesBased\CoherencesDiffs')
% load C3NREMDiff.mat; load C3REMDiff.mat; load F3NREMDiff.mat
% load F3REMDiff.mat; load O1NREMDiff.mat; load O1REMDiff.mat
% load C3Diff.mat; load F3Diff.mat; load O1Diff.mat
% load f_TOT.mat
% fs = 256;

% DiffMatrix = horzcat(C3Diff(1:256, 1), C3NREMDiff(1:256, 1),C3REMDiff(1:256, 1), ...
%         F3Diff(1:256, 1), F3NREMDiff(1:256, 1), F3REMDiff(1:256, 1), ...
%         O1Diff(1:256, 1), O1NREMDiff(1:256, 1), O1REMDiff(1:256, 1));
% group = ["C3", "C3NREM", "C3REM", "F3", "F3NREM", "F3REM", "O1", "O1NREM", "O1REM"];
% for item = 1:3:9
%     figure
%     subplot(3,1,1)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item), 'b')
%     grid on
%     title('Difference of the CPSD of', group(item))
%     subplot(3,1,2)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item+1), 'b')
%     grid on
%     title('Difference of the CPSD of', group(item+1))
%     subplot(3,1,3)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item+2), 'b')
%     grid on
%     title('Difference of the CPSD of', group(item+2))
%     xlabel('frequency (Hz)','FontSize',12)
% end

%% Anova Analysis between the coherences of the groups
folderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\SleepStagesBased\Sleep stages Avg Coherences';
fileList = dir(fullfile(folderpath, '*.mat'));
for i = 1:numel(fileList)
    if i >= 1 && i < 5
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        C3Matrix(:, i) = Matfile.(FileName(1:12));
    elseif i >= 5 && i < 9
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        F3Matrix(:, i-4) = Matfile.(FileName(1:12));
    elseif i > 8
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        O1Matrix(:, i-8) = Matfile.(FileName(1:12));
    end
end
O1Matrix = O1Matrix(1:256, :);
group = ["O1NREMCont", "O1NREMDep", "O1REMCont", "O1REMDep"];
[~,~,stats] = anova1(O1Matrix,group);
figure
[c,~,~,gnames] = multcompare(stats);
