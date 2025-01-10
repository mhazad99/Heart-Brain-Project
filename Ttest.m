clc
clear
close all

%% initials
load f_TOT.mat
fs = 256;
%% load Coherences
Depressionfolderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Insomniac Cases\CoherencesF3';
DepressionfileList = dir(fullfile(Depressionfolderpath, '*.mat'));
Controlfolderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\ControlCoherencesF3';
ControlfileList = dir(fullfile(Controlfolderpath, '*.mat'));
%% read files
MCCont = [];MCDep = [];PCCont = [];PCDep = [];
for i = 1:numel(DepressionfileList)
    ControlFileName = ControlfileList(i).name; % Get the file name
    ControlfilePath = fullfile(Controlfolderpath, ControlFileName); % Get the full file path
    CMatfile = load(ControlfilePath);
    if contains(ControlFileName,'Mag')
        MCCont(:, i) = CMatfile.MagCoherence;
        f_TOT = CMatfile.f_TOT;
    elseif contains(ControlFileName, 'Phase')
        PCCont(:, i) = CMatfile.PhaseCoherence;
        f_TOT = CMatfile.f_TOT;
    end
    DepressionFileName = DepressionfileList(i).name; % Get the file name
    DepressionfilePath = fullfile(Depressionfolderpath, DepressionFileName); % Get the full file path
    DMatfile = load(DepressionfilePath);
    if contains(DepressionFileName,'Mag')
        MCDep(:, i) = DMatfile.MagCoherence;
        f_TOT = DMatfile.f_TOT;
    elseif contains(DepressionFileName, 'Phase')
        PCDep(:, i) = DMatfile.PhaseCoherence;
        f_TOT = DMatfile.f_TOT;
    end    
end
% Control files
Average_CCont = mean(MCCont, 2); STD_CCont = std(MCCont, 0, 2);
Average_PCont = mean(PCCont, 2); STD_PCont = std(PCCont, 0, 2);
% Depression files
Average_CDep = mean(MCDep, 2); STD_CDep = std(MCDep, 0, 2);
Average_PDep = mean(PCDep, 2); STD_PDep = std(PCDep, 0, 2);
%% Depression Cases
figure('Position',[200, 200, 600, 400])
ax(1) = subplot(3,1,1);
area(f_TOT,Average_CDep+STD_CDep,'FaceColor',[0.8 0.8 1])
hold on
area(f_TOT,Average_CDep-STD_CDep,'FaceColor','w')
hold on
plot(f_TOT,Average_CDep,'b','LineWidth',2)
title('Average Magnitude Coherence of Depression Cases', FontSize=12)
legend('STD of Average Magnitude Coherence of Depression Cases','Fontsize', 10)
%legend('Average Magnitude Coherence of Depression Cases','Fontsize', 12)
%grid on

%% Control Cases
ax(2) = subplot(3,1,2);
area(f_TOT,Average_CCont+STD_CCont,'FaceColor',[1 0.8 0.8])
hold on
area(f_TOT,Average_CCont-STD_CCont,'FaceColor','w')
hold on
plot(f_TOT,Average_CCont,'r','LineWidth',2)
%grid on
title('Average Magnitude Coherence of Control Cases', FontSize=12)
legend('STD of Average Magnitude Coherence of Control Cases','Fontsize', 10)
%legend('Average Magnitude Coherence of Control Cases','Fontsize', 12)

WindowLength = (length(f_TOT)-1)/(fs/2);
for item = 1:length(MCCont)
    [h1,p1,~,stats1] = ttest2(MCDep(item, :),MCCont(item, :), 'Vartype','unequal', 'Alpha', 0.05);
    h_Storage_Mag(item, :) = h1;
    p_Storage_Mag(item, :) = p1;
    STDstorage_Mag(item, :) = stats1.sd;
    %DFstorage_Mag(item, :) = stats1.df;
%     [h2,p2,~,stats2] = ttest2(PCDep(item, :), PCCont(item, :), 'Vartype','unequal','Alpha',0.05);
%     h_Storage_Phase(item, :) = h2;
%     p_Storage_Phase(item, :) = p2;
%     STDstorage_Phase(item, :) = stats2.sd;
%     %DFstorage_Phase(item, :) = stats2.df;
end
for item = 1:length(p_Storage_Mag)
    if p_Storage_Mag(item)>0.05
        p_Storage_Mag(item) = 0.05;
    end
%     if p_Storage_Phase(item)>0.05
%         p_Storage_Phase(item) = 0.05;
%     end
end
ax(3) = subplot(3,1,3);
plot(f_TOT, p_Storage_Mag, 'k','LineWidth',1)
grid on

legend('pvalue', 'Fontsize', 10,'Location','Best')
xlabel('frequency (Hz)','FontSize',12)
%%% Standard error of mean
% for item = 1:length(STDstorage_Mag)
%     SEM_Mag(item,1) = sqrt((STDstorage_Mag(item,1)^2/size(MCCont, 2))+ ...
%                             (STDstorage_Mag(item,2)^2/size(MCDep,2)));
% end
% ax(4) = subplot(4,1,4);
% plot(f_TOT, SEM_Mag, 'r', 'LineWidth',1)
% grid on
% legend('Standard Error of Mean', 'Fontsize', 12)
% xlabel('frequency (Hz)','FontSize',20)
linkaxes(ax,'x');
xlim(ax, [0, 35])
ylim(ax, [0.45, 0.75])
ylim(ax(3), [0, 0.055])
zoom on
%%% Standard error of mean
% for item = 1:length(STDstorage_Mag)
%     SEM_Phase(item,1) = sqrt((STDstorage_Phase(item,1)^2/size(PCCont, 2))+ ...
%                             (STDstorage_Phase(item,2)^2/size(PCDep,2)));
% end
%% Area plots for phase
% figure
% az(1) = subplot(3,1,1);
% area(f_TOT,Average_PDep+STD_PDep,'FaceColor',[0.8 0.8 1],'BaseValue',-4)
% hold on
% area(f_TOT,Average_PDep-STD_PDep,'FaceColor','w','BaseValue',-4)
% hold on
% plot(f_TOT,Average_PDep,'b','LineWidth',2)
% %grid on
% title('Average Phase Coherence of Depression Cases')
% legend('STD of Average Phase Coherence of Depression Cases','Fontsize', 12)
% az(2) = subplot(3,1,2);
% area(f_TOT,Average_PCont+STD_PCont,'FaceColor',[1 0.8 0.8],'BaseValue',-4)
% hold on
% area(f_TOT,Average_PCont-STD_PCont,'FaceColor','w','BaseValue',-4)
% hold on
% plot(f_TOT,Average_PCont,'r','LineWidth',2)
% %grid on
% title('Average Phase Coherence of Control Cases')
% legend('STD of Average Phase Coherence of Control Cases','Fontsize', 12)
% az(3) = subplot(3,1,3);
% plot(f_TOT, p_Storage_Phase,'k')
% grid on
% legend('pvalue', 'Fontsize', 12,'Location','Best')
% xlabel('frequency (Hz)','FontSize',20)
% %%% Standard error of mean plot
% % az(4) = subplot(4,1,4);
% % plot(f_TOT, SEM_Phase, 'r', 'LineWidth',1)
% % legend('Standard Error of Mean', 'Fontsize', 12)
% % xlabel('frequency (Hz)','FontSize',20)
% % grid on
% linkaxes(az,'x');
% zoom on
%% Clustering the frequencies
[Magoutputs] = ClusterFreq(f_TOT, Average_CDep, Average_CCont, MCCont, MCDep, true);
disp(Magoutputs)
%%% To see the results for the phase, uncomment next line!
%[Phaseoutputs] = ClusterFreq(f_TOT, Average_PDep, Average_PCont, PCCont, PCDep, false);

% for item = 1:length(ShuffledArray)
%     [hB,pB,~,statsB] = ttest2(ShuffledArray(item, 1:l/2),ShuffledArray(item, (l/2)+1:l), ...
%         'Vartype','unequal', 'Alpha', 0.1);
%     h_Storage_MagB(item, :) = hB;
%     p_Storage_MagB(item, :) = pB;
%     STDstorage_MagB(item, :) = statsB.sd;
%     %DFstorage_Mag(item, :) = statsB.df;
% end
% for item = 1:length(STDstorage_MagB)
%     SEM_MagB(item,1) = sqrt((STDstorage_MagB(item,1)^2/size(MCCont, 2))+ ...
%                             (STDstorage_MagB(item,2)^2/size(MCDep,2)));
% end
% for item = 1:length(p_Storage_MagB)
%     if p_Storage_MagB(item)>0.1
%         p_Storage_MagB(item) = 0.1;
%     end
% end
% figure
% ay(1) = subplot(3,1,1);
% plot(f_TOT, FirstGPaverage, 'b')
% hold on
% plot(f_TOT, SecondGPaverage, 'r')
% grid on
% legend('Average Magnitude Coherence of first random gp', ['Average ' ...
%     'Magnitude Coherence of second random gp'],'Fontsize', 12)
% ay(2) = subplot(3,1,2);
% plot(f_TOT, p_Storage_MagB,'k')
% grid on
% legend('pvalue', 'Fontsize', 12,'Location','Best')
% ay(3) = subplot(3,1,3);
% plot(f_TOT, SEM_MagB, 'r', 'LineWidth',1)
% legend('Standard Error of Mean', 'Fontsize', 12)
% xlabel('frequency (Hz)','FontSize',20)
% grid on
% linkaxes(ay,'x');
% zoom on
%% Bootstrapping for Phase to make sure the results are reliable
% l = size(PCCont,2) + size(PCDep,2);
% pooledArrayPh = [PCCont, PCDep];
% randidx = randperm(l);
% ShuffledArrayPh = pooledArrayPh(:, randidx);
% for item = 1:length(ShuffledArrayPh)
%     [hB,pB,~,statsB] = ttest2(ShuffledArrayPh(item, 1:l/2),ShuffledArrayPh(item, (l/2)+1:l), ...
%         'Vartype','unequal', 'Alpha', 0.1);
%     h_Storage_PhB(item, :) = hB;
%     p_Storage_PhB(item, :) = pB;
%     STDstorage_PhB(item, :) = statsB.sd;
%     %DFstorage_Mag(item, :) = statsB.df;
% end
% for item = 1:length(STDstorage_PhB)
%     SEM_PhB(item,1) = sqrt((STDstorage_PhB(item,1)^2/size(PCCont, 2))+ ...
%                             (STDstorage_PhB(item,2)^2/size(PCDep,2)));
% end
% for item = 1:length(p_Storage_PhB)
%     if p_Storage_PhB(item)>0.1
%         p_Storage_PhB(item) = 0.1;
%     end
% end
% FirstGPaverage1 = mean(ShuffledArrayPh(:,1:l/2),2);
% SecondGPaverage1 = mean(ShuffledArrayPh(:,(l/2)+1:l),2);
% figure
% ay(1) = subplot(3,1,1);
% plot(f_TOT, FirstGPaverage1, 'b')
% hold on
% plot(f_TOT, SecondGPaverage1, 'r')
% grid on
% legend('Average Magnitude Coherence of first random gp', ['Average ' ...
%     'Magnitude Coherence of second random gp'],'Fontsize', 12)
% ay(2) = subplot(3,1,2);
% plot(f_TOT, p_Storage_PhB,'k')
% grid on
% legend('pvalue', 'Fontsize', 12,'Location','Best')
% ay(3) = subplot(3,1,3);
% plot(f_TOT, SEM_PhB, 'r', 'LineWidth',1)
% legend('Standard Error of Mean', 'Fontsize', 12)
% xlabel('frequency (Hz)','FontSize',20)
% grid on
% linkaxes(ay,'x');
% zoom on

% %% MAE for Magnitude of coherence (actual and bootstrapped group)
% absolute_error1 = abs(SEM_MagB - SEM_Mag);
% MAE_Mag = mean(absolute_error1);
% %% MAE for Phase of coherence (actual and bootstrapped group)
% absolute_error = abs(SEM_PhB - SEM_Phase);
% MAE_Phase = mean(absolute_error);