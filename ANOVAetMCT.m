clc
clear
close all
%% To plot the differences of coherences for sleep stages
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Insomniac Cases\DiffCoherences')
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Insomniac Cases\AvgCoherences')
% load C3NREMDiff.mat; load C3REMDiff.mat; load F3NREMDiff.mat
% load F3REMDiff.mat; load O1NREMDiff.mat; load O1REMDiff.mat
load C3Diff.mat; load F3Diff.mat; load O1Diff.mat
load C3AvgCon.mat; load F3AvgCon.mat; load O1AvgCon.mat;
load C3AvgDep.mat; load F3AvgDep.mat; load O1AvgDep.mat;
load f_TOT.mat
fs = 256;
% ControlCoherences = [C3AvgCon(1:256, 1), F3AvgCon(1:256, 1), O1AvgCon(1:256, 1)];
% DepressionCoherences = [C3AvgDep(1:256, 1), F3AvgDep(1:256, 1), O1AvgDep(1:256, 1)];
% gps = ["C3Avg", "F3Avg", "O1Avg"];
% [~,~,stats] = anova1(ControlCoherences, gps);
% figure
% [c,~,~,gnames] = multcompare(stats);

Difs = [C3Diff(1:256, 1), F3Diff(1:256, 1), O1Diff(1:256, 1)];
%%%%% Uncomment when you are using the sleep stage based analysis
% DiffMatrix = horzcat(C3Diff(1:256, 1), C3NREMDiff(1:256, 1),C3REMDiff(1:256, 1), ...
%         F3Diff(1:256, 1), F3NREMDiff(1:256, 1), F3REMDiff(1:256, 1), ...
%         O1Diff(1:256, 1), O1NREMDiff(1:256, 1), O1REMDiff(1:256, 1));
%O1group = ["C3", "C3NREM", "C3REM", "F3", "F3NREM", "F3REM", "O1", "O1NREM", "O1REM"];
Channelgroup = ["C3", "F3", "O1"];

%%%%%%%% For the entire night analysis
for item = 1:1
    figure
    subplot(3,1,1)
    plot(f_TOT(1:256, 1),Difs(:, item), 'b')
    grid on
    title('Difference of the CPSD of', Channelgroup(item))
    xlim([0, 8]);
    ylim([-0.016, 0.005]);
    subplot(3,1,2)
    plot(f_TOT(1:256, 1),Difs(:, item+1), 'b')
    grid on
    title('Difference of the CPSD of', Channelgroup(item+1))
    xlim([0, 8]);
    ylim([-0.016, 0.005]);
    subplot(3,1,3)
    plot(f_TOT(1:256, 1),Difs(:, item+2), 'b')
    grid on
    title('Difference of the CPSD of', Channelgroup(item+2))
    xlim([0, 8]);
    ylim([-0.016, 0.005]);
    xlabel('frequency (Hz)','FontSize',12)
end

%%%%%%%%% For sleep stage based analysis
% for item = 1:3:9
%     figure
%     subplot(3,1,1)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item), 'b')
%     grid on
%     title('Difference of the CPSD of', O1group(item))
%     xlim([0, 8]);
%     ylim([-0.016, 0.005]);
%     subplot(3,1,2)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item+1), 'b')
%     grid on
%     title('Difference of the CPSD of', O1group(item+1))
%     xlim([0, 8]);
%     ylim([-0.016, 0.005]);
%     subplot(3,1,3)
%     plot(f_TOT(1:256, 1),DiffMatrix(:, item+2), 'b')
%     grid on
%     title('Difference of the CPSD of', O1group(item+2))
%     xlim([0, 8]);
%     ylim([-0.016, 0.005]);
%     xlabel('frequency (Hz)','FontSize',12)
% end

%% Anova Analysis between the coherences of the groups
folderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Insomniac Cases\AvgCoherences';
fileList = dir(fullfile(folderpath, '*.mat'));

%%%%%%% For sleep stages based analysis
% for i = 1:numel(fileList)
%     if i >= 1 && i < 5
%         FileName = fileList(i).name; % Get the file name
%         filePath = fullfile(folderpath, FileName); % Get the full file path
%         Matfile = load(filePath);
%         C3Matrix(:, i) = Matfile.(FileName(1:12));
%     elseif i >= 5 && i < 9
%         FileName = fileList(i).name; % Get the file name
%         filePath = fullfile(folderpath, FileName); % Get the full file path
%         Matfile = load(filePath);
%         F3Matrix(:, i-4) = Matfile.(FileName(1:12));
%     elseif i > 8
%         FileName = fileList(i).name; % Get the file name
%         filePath = fullfile(folderpath, FileName); % Get the full file path
%         Matfile = load(filePath);
%         O1Matrix(:, i-8) = Matfile.(FileName(1:12));
%     end
% end
% C3Matrix = C3Matrix(1:256, :);
% F3Matrix = F3Matrix(1:256, :);
% O1Matrix = O1Matrix(1:256, :);
% Allchannels = [C3Matrix, F3Matrix, O1Matrix];
% O1group = ["O1NREMCont", "O1NREMDep", "O1REMCont", "O1REMDep"];
% C3group = ["C3NREMCont", "C3NREMDep", "C3REMCont", "C3REMDep"];
% F3group = ["F3NREMCont", "F3NREMDep", "F3REMCont", "F3REMDep"];
% groups = ["C3NREMCont", "C3NREMDep", "C3REMCont", "C3REMDep",...
%           "F3NREMCont", "F3NREMDep", "F3REMCont", "F3REMDep",...
%           "O1NREMCont", "O1NREMDep", "O1REMCont", "O1REMDep"];
% %[~,~,stats] = anova2(C3Matrix,1, 'on');
% NREMs = [C3Matrix(:, 1), F3Matrix(:, 1), O1Matrix(:, 1)];
% NREMsGp = ["C3NREMCont", "F3NREMCont", "O1NREMCont"];
% [~,~,stats] = anova1(Allchannels, groups, 'alpha', 0.01);
% figure
% [c,~,~,gnames] = multcompare(stats);


%%%%%%% For entire night analysis
for i = 1:numel(fileList)
    if i >= 1 && i < 3
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        C3Matrix(:, i) = Matfile.(FileName(1:8));
    elseif i >= 3 && i < 5
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        F3Matrix(:, i-2) = Matfile.(FileName(1:8));
    elseif i > 4
        FileName = fileList(i).name; % Get the file name
        filePath = fullfile(folderpath, FileName); % Get the full file path
        Matfile = load(filePath);
        O1Matrix(:, i-4) = Matfile.(FileName(1:8));
    end
end
C3Matrix = C3Matrix(16:32, :);
F3Matrix = F3Matrix(16:32, :);
O1Matrix = O1Matrix(16:32, :);
Allchannels = [C3Matrix, F3Matrix, O1Matrix];
Channelgroup = ["O1NREMCont", "O1NREMDep", "O1REMCont", "O1REMDep"];
C3group = ["C3NREMCont", "C3NREMDep", "C3REMCont", "C3REMDep"];
F3group = ["F3NREMCont", "F3NREMDep", "F3REMCont", "F3REMDep"];
groups = ["C3NREMCont", "C3NREMDep", "C3REMCont", "C3REMDep",...
          "F3NREMCont", "F3NREMDep", "F3REMCont", "F3REMDep",...
          "O1NREMCont", "O1NREMDep", "O1REMCont", "O1REMDep"];
Channelgroups = ["C3Cont", "C3Insm", "F3Cont", "F3Insm", "O1Cont", "O1Insm"];
%[~,~,stats] = anova2(C3Matrix,1, 'on');
NREMs = [C3Matrix(:, 1), F3Matrix(:, 1), O1Matrix(:, 1)];
NREMsGp = ["C3NREMCont", "F3NREMCont", "O1NREMCont"];
[~,~,stats] = anova1(Allchannels, Channelgroups, 'alpha', 0.01);
figure
[c,~,~,gnames] = multcompare(stats);
