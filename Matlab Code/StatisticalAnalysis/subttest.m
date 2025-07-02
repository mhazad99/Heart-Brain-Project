function [outputs] = subttest(MaximumInClusterboot, MaximumInCluster, flag)
%% first cluster
[hb1,pb1,ci1,statsb1] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(1), 'Alpha',0.01);
outputs.ci1 = ci1;
outputs.pb1 = pb1;
outputs.hb1 = hb1;
outputs.tval1 = statsb1.tstat;
outputs.stats1 = statsb1;
nu1 = statsb1.df;
k = linspace(-100,50,300);
tdistpdf = tpdf(k,nu1);
tval = statsb1.tstat;
tvalpdf = tpdf(tval,nu1);
tcrit = tinv(0.99,nu1);
figure('Position',[200, 200, 600, 400])
subplot(4,2,1)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('First cluster')
%if flag == false
%% second cluster
[hb2,pb2,ci2,statsb2] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(2), 'Alpha',0.01);
outputs.ci2 = ci2;
outputs.pb2 = pb2;
outputs.hb2 = hb2;
outputs.tval2 = statsb2.tstat;
nu2 = statsb2.df;
k = linspace(-100,50,300);
tdistpdf = tpdf(k,nu2);
tval = statsb2.tstat;
tvalpdf = tpdf(tval,nu2);
tcrit = tinv(0.99,nu2);
subplot(4,2,2)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('Second cluster')
%% third cluster
[hb3,pb3,ci3,statsb3] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(3), 'Alpha',0.01);
outputs.ci3 = ci3;
outputs.pb3 = pb3;
outputs.hb3 = hb3;
outputs.tval3 = statsb3.tstat;
nu3 = statsb3.df;
k = linspace(-100,50,300);
tdistpdf = tpdf(k,nu3);
tval = statsb3.tstat;
tvalpdf = tpdf(tval,nu3);
tcrit = tinv(0.99,nu3);
subplot(4,2,3)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('Third cluster')
%% fourth cluster
[hb4,pb4,ci4,statsb4] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(4), 'Alpha',0.01);
outputs.ci4 = ci4;
outputs.pb4 = pb4;
outputs.hb4 = hb4;
outputs.tval4 = statsb4.tstat;
nu4 = statsb4.df;
k = linspace(-100,50,300);
tdistpdf = tpdf(k,nu4);
tval = statsb4.tstat;
tvalpdf = tpdf(tval,nu4);
tcrit = tinv(0.99,nu4);
subplot(4,2,4)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('Fourth cluster')
%% fifth cluster
[hb5,pb5,ci5,statsb5] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(5), 'Alpha',0.01);
outputs.ci5 = ci5;
outputs.pb5 = pb5;
outputs.hb5 = hb5;
outputs.tval5 = statsb5.tstat;
nu5 = statsb5.df;
k = linspace(-100,60,300);
tdistpdf = tpdf(k,nu5);
tval = statsb5.tstat;
tvalpdf = tpdf(tval,nu5);
tcrit = tinv(0.99,nu5);
subplot(4,2,5)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('Fifth cluster')

%% sixth cluster
[hb6,pb6,ci6,statsb6] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(6), 'Alpha',0.01);
outputs.ci6 = ci6;
outputs.pb6 = pb6;
outputs.hb6 = hb6;
outputs.tval6 = statsb6.tstat;
nu6 = statsb6.df;
k = linspace(-100,60,300);
tdistpdf = tpdf(k,nu6);
tval = statsb6.tstat;
tvalpdf = tpdf(tval,nu6);
tcrit = tinv(0.99,nu6);
subplot(4,2,6)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('Sixth cluster')

%% bar plot
pvals = [pb1, pb2, pb3, pb4, pb5, pb6];
figure;
bar(pvals)
xticklabels({'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'})
title('Statistical Significance of t-Test Results')
xlabel('Comparison Groups')
ylabel('p-Value')
hold on
yline(0.01, 'r--', 'Significance Threshold (alpha = 0.01)')
legend('p-Value', 'Location', 'NorthEast')
%% seventh cluster
[hb7,pb7,ci7,statsb7] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(7), 'Alpha',0.01);
outputs.ci7 = ci7;
outputs.pb7 = pb7;
outputs.hb7 = hb7;
outputs.tval7 = statsb7.tstat;
nu7 = statsb7.df;
k = linspace(-100,60,300);
tdistpdf = tpdf(k,nu7);
tval = statsb6.tstat;
tvalpdf = tpdf(tval,nu7);
tcrit = tinv(0.99,nu7);
subplot(4,2,7)
plot(k,tdistpdf)
hold on
scatter(tval,tvalpdf,"filled")
xline(tcrit,"--")
xline(-tcrit,"--")
legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
title('T-test resutls for the cluster considering seventh min(0-8Hz & confidence level of 0.99)')

% %% eighth cluster
% [hb8,pb8,ci8,statsb8] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(8), 'Alpha',0.01);
% outputs.ci8 = ci8;
% outputs.pb8 = pb8;
% outputs.hb8 = hb8;
% outputs.tval8 = statsb8.tstat;
% nu8 = statsb8.df;
% k = linspace(-60,60,300);
% tdistpdf = tpdf(k,nu8);
% tval = statsb8.tstat;
% tvalpdf = tpdf(tval,nu8);
% tcrit = tinv(0.99,nu8);
% subplot(4,2,8)
% plot(k,tdistpdf)
% hold on
% scatter(tval,tvalpdf,"filled")
% xline(tcrit,"--")
% xline(-tcrit,"--")
% legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
% title('T-test resutls for the cluster considering eigth max(0-8Hz & confidence level of 0.99)')

if flag == false
    %% seventh cluster
    [hb7,pb7,ci7,statsb7] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(7), 'Alpha',0.05);
    outputs.ci7 = ci7;
    outputs.pb7 = pb7;
    outputs.hb7 = hb7;
    outputs.tval7 = statsb7.tstat;
    nu7 = statsb7.df;
    k = linspace(-60,60,300);
    tdistpdf = tpdf(k,nu7);
    tval = statsb7.tstat;
    tvalpdf = tpdf(tval,nu7);
    tcrit = tinv(0.95,nu7);
    subplot(4,2,7)
    plot(k,tdistpdf)
    hold on
    scatter(tval,tvalpdf,"filled")
    xline(tcrit,"--")
    xline(-tcrit,"--")
    legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
    title('T-test resutls for the cluster considering seventh max(0-8Hz & confidence level of 0.95)')
    %% eight cluster
    [hb8,pb8,ci8,statsb8] = ttest(MaximumInClusterboot(:, 1), MaximumInCluster(7), 'Alpha',0.05);
    outputs.ci7 = ci8;
    outputs.pb7 = pb8;
    outputs.hb7 = hb8;
    outputs.tval7 = statsb8.tstat;
    nu8 = statsb8.df;
    k = linspace(-60,60,300);
    tdistpdf = tpdf(k,nu8);
    tval = statsb8.tstat;
    tvalpdf = tpdf(tval,nu8);
    tcrit = tinv(0.95,nu8);
    subplot(4,2,8)
    plot(k,tdistpdf)
    hold on
    scatter(tval,tvalpdf,"filled")
    xline(tcrit,"--")
    xline(-tcrit,"--")
    legend(["Student's t pdf", "t-Statistic", "Critical Cutoff"])
    title('T-test resutls for the cluster considering eight max(0-8Hz & confidence level of 0.95)')
    if flag == false
    end
end