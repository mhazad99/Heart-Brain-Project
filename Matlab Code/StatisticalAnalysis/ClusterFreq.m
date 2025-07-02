function [Clusteroutputs] = ClusterFreq(f_TOT, AverageDep, AverageCont, CCont, CDep, flag)
% if flag = true : Magnitude elseif flage = false : Phase
if flag
    % In this case it calculates the method for magnitude
    Diff = AverageDep - AverageCont;
    % Apply the moving average filter on the difference to make it smoother
    window_size = 10;
    Diff = movmean(Diff, window_size);
    %%% Saving the Diffs in another path
    Path = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Insomniac Cases\AvgCoherences';
    F3AvgCon = AverageCont;
    F3AvgDep = AverageDep;
    fileName1 = 'F3AvgCon.mat';
    fullFilePath1 = fullfile(Path, fileName1);
    fileName2 = 'F3AvgDep.mat';
    fullFilePath2 = fullfile(Path, fileName2);
    F3Diff = Diff;
    fileName3 = 'F3Diff.mat';
    fullFilePath3 = fullfile(Path, fileName3);
    save(fullFilePath1, 'F3AvgCon')
    save(fullFilePath2, 'F3AvgDep')
    save(fullFilePath3, "F3Diff")
    %%%
    %Diff = abs(Diff);
    flagg = true; % If you want to find the peaks (true), otherwise: false
    for item = 1:1 % Set it based on the cluster frequency threshold (Visually!)
        if flagg
            maxcluster = max(Diff(1+256*(item-1):256*item));% 256 is defined
            % based on the threshold for cluster frequency for a cluster with
            % 8Hz length (each frequency consists of 32 samples)
            [pks,locs] = findpeaks(Diff(1:256),'MinPeakDistance',30); % for 8Hz cluster
            Maxindex = find(Diff == maxcluster);
            %MaximumInCluster(item) = maxcluster;
            %Maxindex(item) = maxindex;
            for i = 1: length(pks)
                if pks(i) >= 0.005
                    MaximumInCluster(i) = pks(i);
                    Maxindex(i) = locs(i);
                end
            end
        else
            mincluster = min(Diff(1+256*(item-1):256*item));
            [troughs,locs] = findpeaks(-Diff(1:256),'MinPeakDistance',55); % for 8Hz cluster
            minindex = find(Diff == mincluster);
            for i = 1: length(troughs)
                if  troughs(i) >= 0.005
                    MinimumInCluster(i) = troughs(i);
                    minindex(i) = locs(i);
                end
            end
            indices1 = MinimumInCluster == 0;
            indices2 = minindex == 0;
            MinimumInCluster(indices1) = [];
            minindex(indices2) = [];
        end

    end
    %     %%%% since we want to find exact clusters we should check it visually
    %     %%%% to make sure the start and end point of each cluster!
    %     % Finding the first cluster's info
    %     maxcluster = max(Diff(11:45)); % First cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(1) = maxcluster;
    %     Maxindex(1) = maxindex;
    %     % Finding the second cluster's info
    %     maxcluster = max(Diff(45:70)); % Second cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(2) = maxcluster;
    %     Maxindex(2) = maxindex;
    %     % Finding the third cluster's info
    %     maxcluster = max(Diff(71:100)); % Third cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(3) = maxcluster;
    %     Maxindex(3) = maxindex;
    %     % Finding the fourth cluster's info
    %     maxcluster = max(Diff(108:137)); % Fourth cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(4) = maxcluster;
    %     Maxindex(4) = maxindex;
    %     % Finding the fifth cluster's info
    %     maxcluster = max(Diff(137:208)); % Fifth cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(5) = maxcluster;
    %     Maxindex(5) = maxindex;
    %     % Finding the fifth cluster's info
    %     maxcluster = max(Diff(208:251)); % Fifth cluster
    %     maxindex = find(Diff == maxcluster);
    %     MaximumInCluster(6) = maxcluster;
    %     Maxindex(6) = maxindex;

    figure('Position',[200, 200, 600, 400])
    title(['Cluster the difference between Magnitude of coherence of two groups' ...
        ' based on the fixed thresholds'], 'FontSize',15)
    for ploti = 1:length(MaximumInCluster)                                  %%% Change Maximumincluster and minimumincluster for peaks and troughs respectively.
        ax(ploti) = subplot(4, 2, ploti);
        plot(f_TOT, Diff, 'b')                                              %Change minus to plus for troughs and peaks respectively
        hold on
        scatter(f_TOT(Maxindex(ploti)), MaximumInCluster(ploti),'filled','MarkerFaceColor','r') %%% change it for troughs and peaks
        yline(0.005, LineWidth=2)                                           % Threshold to fix the number of cluster frequencies
        xline(8, LineWidth=1)
        legend('Difference between magnitude of coherence of two groups', 'Maximum in cluster', 'fontsize', 12)
        grid on
        xlabel('frequency (Hz)','FontSize',8)
        linkaxes(ax,'x');
        xlim(ax, [0, 9])
        zoom on
    end

    %     xline(1.3475, LineWidth=1);xline(2.15625, LineWidth=1); % These thresholds have been fixed based on the clusters
    %     xline(3.09375, LineWidth=1);xline(3.34375, LineWidth=1);
    %     xline(4.25, LineWidth=1);xline(6.46875, LineWidth=1);
    %     xline(7.8125, LineWidth=1);


    %% Bootstrapping for Magnitude to make sure the results are reliable

    l = size(CCont,2) + size(CDep,2);
    pooledArrayMag = [CCont, CDep];
    for iter = 1:1500 % number of bootstrappings
        %randidx = randperm(l);
        for random = 1:l
            randidx = randi(l);
            randidxarray(1, random) = randidx;
        end
        ShuffledArray = pooledArrayMag(:, randidxarray);
        FirstGPaverage = mean(ShuffledArray(:,1:l/2),2);
        SecondGPaverage = mean(ShuffledArray(:,(l/2)+1:l),2);
        BootDiff = SecondGPaverage - FirstGPaverage;
        %MagBootDiff = abs(BootDiff);
        window_size = 10;
        BootDiff = movmean(BootDiff, window_size);
        for item = 1:1 % number of clusters                                 %%% Change the min to max in case needed if you switched to troughs instead of peaks(ex. O1)
            maxcluster = max(BootDiff(1+256*(item-1):256*item)); % 256 is defined
            % based on the threshold for cluster frequency (Defines the 8 Hz cluster, you can push it if you want go further than 8 Hz)
            if maxcluster >= 0.005
                MaximumInClusterboot(iter, item) = maxcluster;
            end
        end

        %     %%%% since we want to find exact clusters we should check it visually
        %     %%%% to make sure the start and end point of each cluster!
        %     % Finding the first cluster's info
        %     maxcluster = max(BootDiff(11:45)); % First cluster
        %     MaximumInClusterboot(iter, 1) = maxcluster;
        %     % Finding the second cluster's info
        %     maxcluster = max(BootDiff(45:70)); % Second cluster
        %     MaximumInClusterboot(iter, 2) = maxcluster;
        %     % Finding the third cluster's info
        %     maxcluster = max(BootDiff(71:100)); % Third cluster
        %     MaximumInClusterboot(iter, 3) = maxcluster;
        %     % Finding the fourth cluster's info
        %     maxcluster = max(BootDiff(108:137)); % Fourth cluster
        %     MaximumInClusterboot(iter, 4) = maxcluster;
        %     % Finding the fifth cluster's info
        %     maxcluster = max(BootDiff(137:208)); % Fifth cluster
        %     MaximumInClusterboot(iter, 5) = maxcluster;
        %     % Finding the fifth cluster's info
        %     maxcluster = max(BootDiff(208:251)); % Fifth cluster
        %     MaximumInClusterboot(iter, 6) = maxcluster;

    end
    indx1 = MaximumInClusterboot == 0;
    MaximumInClusterboot(indx1) = [];

    % Calculate the t-test for each bootstrapp cluster
    %Plots
    figure
    %     Stringlist1 = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'};
    %     Stringlist2 = {'(0.31-1.34Hz)', '(1.34-2.15Hz)', '(2.15-3.09Hz)', '(3.34-4.25Hz)', '(4.25-6.46Hz)', '(6.46-7.81Hz)'};
    %     for ploti = 1:6
    %         subplot(3,2,ploti)
    %         histfit(MaximumInClusterboot(:, ploti))
    %         title(sprintf('Maximum values of %s cluster %s for 1000 times bootstrapping' ...
    %             , string(Stringlist1(ploti)), string(Stringlist2(ploti))))
    %         xlabel("Maximum value")
    %         ylabel("Frequency")
    %     end
    Clusteroutputs.MaximumInClusterboot = MaximumInClusterboot(:, 1);
    Clusteroutputs.AvgMaximumInClusterboot = mean(MaximumInClusterboot(:, 1));
    figure('Position',[200, 200, 600, 400])
    histfit(MaximumInClusterboot(:, 1))
    title('Maximum values of the cluster for 1000 times bootstrapping');
    xlabel("Maximum value")
    ylabel("Frequency")
    % Ttest for each cluster
    [outputs] = subttest(MaximumInClusterboot, MaximumInCluster, flag);     % change the MinimumInCluster to MaximumInCluster in case needed
    Clusteroutputs.subttestoutputs = outputs;
    Clusteroutputs.MaximumInCluster = MaximumInCluster;                     % change the MinimumInCluster to MaximumInCluster in case needed
else
    % In this case it calculates the method for phase
    Diff = AverageDep - AverageCont;
    Diff = abs(Diff);
    for item = 1:7 % Set it based on the cluster frequency threshold (Visually!)
        maxcluster = max(Diff(1+64*(item-1):64*item));% 64 is defined
        % based on the threshold for cluster frequency
        maxindex = find(Diff == maxcluster);
        MaximumInCluster(item) = maxcluster;
        Maxindex(item) = maxindex;
    end
    figure
    plot(f_TOT, Diff, 'b')
    hold on
    for ploti = 1:7
        scatter(f_TOT(Maxindex(ploti)), MaximumInCluster(ploti),'filled','MarkerFaceColor','r')
    end
    yline(0.15, LineWidth=2) % Threshold to fix the number of cluster frequencies
    xline(2, LineWidth=1);xline(4, LineWidth=1);xline(6, LineWidth=1);
    xline(8, LineWidth=1);xline(10, LineWidth=1);xline(12, LineWidth=1);xline(14, LineWidth=1)
    legend('Difference between phase of coherence of two groups', 'Maximum in each cluster')
    grid on
    xlabel('frequency (Hz)','FontSize',15)
    title(['Cluster the difference between phase of coherence of two groups' ...
        ' based on the fixed thresholds'], 'FontSize',15)
    %% Bootstrapping for Phase to make sure the results are reliable
    l = size(CCont,2) + size(CDep,2);
    pooledArrayPh = [CCont, CDep];
    for iter = 1:1000 % number of bootstrappings
        %randidx = randperm(l);
        for random = 1:l
            randidx = randi(l);
            randidxarray(1, random) = randidx;
        end
        ShuffledArray = pooledArrayPh(:, randidxarray);
        FirstGPaverage = mean(ShuffledArray(:,1:l/2),2);
        SecondGPaverage = mean(ShuffledArray(:,(l/2)+1:l),2);
        BootDiff = SecondGPaverage - FirstGPaverage;
        MagBootDiff = abs(BootDiff);
        for item = 1:7 % number of clusters
            maxcluster = max(MagBootDiff(1+64*(item-1):64*item)); % 64 is defined
            % based on the threshold for cluster frequency
            MaximumInClusterboot(iter, item) = maxcluster;
        end
    end
    % Calculate the t-test for each bootstrapp cluster
    %Plots
    figure
    Stringlist1 = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'};
    Stringlist2 = {'(0-2Hz)', '(2-4Hz)', '(4-6Hz)', '(6-8Hz)', '(8-10Hz)', '(10-12Hz)', '(12-14Hz)'};
    for ploti = 1:7
        subplot(4,2,ploti)
        histfit(MaximumInClusterboot(:, ploti))
        title(sprintf('Maximum values of %s cluster %s for 1000 times bootstrapping' ...
            , string(Stringlist1(ploti)), string(Stringlist2(ploti))))
        xlabel("Maximum value")
        ylabel("Frequency")
    end
    % Ttest for each cluster
    [outputs] = subttest(MaximumInClusterboot, MaximumInCluster, flag);

end