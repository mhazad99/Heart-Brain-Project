function [rrTimeSeries] = RRsPostProcessing(mergedQrsCandidates)
%RRSPOSTPROCESSING 
%
% Implementation of an adaptive filtering algorithm. The main advantage of
% this algorithm is the spontaneous adaptation of the filter coefficients
% due to suden changes in the ibi series.
%
% Wessel N., Voss A., Malberg H., Ziehmann C., Voss H. U., Schirdewan A.,
% Meyerfeldt, U., Kurths J., Nonlinear analysis of complex phenomena in
% cardiological data, Herzschr Elektrophys 11:159-173 (2000).
%
global POST_PROCESSING
global GRAPHICS

fprintf('\tPost-Processing of RR-intervals Time Series ...\n');

RRn = mergedQrsCandidates.rrIntervals;
qrsTimeStamps  = mergedQrsCandidates.qrsTimeStamps;

%% ********** 1- Removal of obvious recognition errors **********
errorsIdx = find(RRn > POST_PROCESSING.IBI_PHYSIOLOGICAL_UPPER_LIMIT | ...
                 RRn < POST_PROCESSING.IBI_PHYSIOLOGICAL_LOWER_LIMIT);
RRn(errorsIdx) = [];
qrsTimeStamps(errorsIdx+1) = [];
nbRRn = length(RRn);

%% ********** 2- Adaptive percent-filter **********

%% ---------- 2.A- Create binomial-7 filtered series ----------
h = [1.0 6.0 15.0 20.0 15.0 6.0 1.0]/64.0;
tn = conv(RRn,h,'same');

%% ---------- 2.B- Compute adaptive mean and std of binomial-7 filtered series ----------
mu_a = double.empty;
sig_a = double.empty;
lbd_a = double.empty;
tn2 = tn.^2;
c = POST_PROCESSING.CONTROLLING_COEFFICIENT;
one_minus_c = 1 - c;

mu_a(1) = 0.5*(RRn(1) + tn(1));
lbd_a(1) = RRn(1)^2 - tn2(1);
sig_a(1) = sqrt(abs(mu_a(1)^2 - lbd_a(1)));
for i=2:nbRRn
%     mu_a(i)  = one_minus_c*mu_a(i-1) + c*tn(i-1);
%     lbd_a(i) = one_minus_c*lbd_a(i-1) + c*tn2(i-1);
    mu_a(i)  = c*(tn(i-1) - mu_a(i-1)) +  mu_a(i-1);
    lbd_a(i) = c*(tn2(i-1) - lbd_a(i-1)) + lbd_a(i-1);
    sig_a(i) = sqrt(abs(mu_a(i)^2 - lbd_a(i)));
end    

%% ---------- 2.C- Exclusion rule of filter ----------
p = POST_PROCESSING.PROPORTION_LIMIT;
sig_a_avg = mean(sig_a);
Threshold1 = p*RRn(2:end) + 3*sig_a_avg;
RRn_diff = abs(diff(RRn));
not_normal_idx1 = find(RRn_diff > Threshold1);

not_normal_idx2 =  int32.empty;
RRlv = median(tn);

j = 0;
if ((RRn(1) < RRlv - 3*sig_a_avg) || (RRn(1) > RRlv + 3*sig_a_avg))
    j = 1;
    not_normal_idx2(j) = 1;
end    

nbNotNormal1 = length(not_normal_idx1);
for i=1:nbNotNormal1
    % Find the last valid RR value
    lvIdx = find(RRn_diff(1:not_normal_idx1(i)) <= Threshold1(1:not_normal_idx1(i)), 1, 'last');
    if ~isempty(lvIdx)
        RRlv = RRn(lvIdx(1)+1);
    else
        RRlv = median(tn);
    end   
    
    Threshold2 = p*RRlv + 3*sig_a_avg;
    if (abs(RRn(not_normal_idx1(i)+1)-RRlv) > Threshold2)
        j = j + 1;
        not_normal_idx2(j) = not_normal_idx1(i)+1;
    end    
end

%% ---------- 2.D- Replacement of not normal RR-intervals ----------
nbNotNormal2 = length(not_normal_idx2);
RRnB = RRn;
RRnB(not_normal_idx2) = mu_a(not_normal_idx2) - ...
                       0.5*sig_a(not_normal_idx2) + ...
                       sig_a(not_normal_idx2).*rand(1,nbNotNormal2);

%% ********** 3- Adaptive controlling filter **********

%% ---------- 3.A- Create binomial-7 filtered series ----------
tn = conv(RRnB,h,'same');

%% ---------- 3.B- Compute adaptive mean and std of binomial-7 filtered series ----------
mu_a = double.empty;
sig_a = double.empty;
lbd_a = double.empty;
tn2 = tn.^2;

mu_a(1) = 0.5*(RRnB(1) + tn(1));
lbd_a(1) = RRnB(1)^2 - tn2(1);
sig_a(1) = sqrt(abs(mu_a(1)^2 - lbd_a(1)));
for i=2:nbRRn
%     mu_a(i)  = one_minus_c*mu_a(i-1) + c*tn(i-1);
%     lbd_a(i) = one_minus_c*lbd_a(i-1) + c*tn2(i-1);  
    mu_a(i)  = c*(tn(i-1) - mu_a(i-1)) +  mu_a(i-1);
    lbd_a(i) = c*(tn2(i-1) - lbd_a(i-1)) + lbd_a(i-1);
    sig_a(i) = sqrt(abs(mu_a(i)^2 - lbd_a(i)));
end   

%% ---------- 3.C- Exclusion rule of filter ----------
sig_b = POST_PROCESSING.BASIC_VARIABILITY;
not_normal_idx3 = find(abs(RRnB - mu_a) > (3*sig_a+ sig_b));

%% ---------- 3.D- Replacement of not normal RR-intervals ----------
RRnC = RRnB;
RRnC(not_normal_idx3) = tn(not_normal_idx3);
% nbNotNormal3 = length(not_normal_idx3);
% RRnC(not_normal_idx3) = mu_a(not_normal_idx3) - ...
%                        0.5*sig_a(not_normal_idx3) + ...
%                        sig_a(not_normal_idx3).*rand(1,nbNotNormal3);

% Results
rrTimeSeries.qrsTimeStamps = qrsTimeStamps; % Time from start in seconds
rrTimeSeries.rrIntervals = RRnC; % Ibis in seconds
rrTimeSeries.rrHR = 60.0./RRnC; % Heart rate in bpm
IBIs2 = diff(qrsTimeStamps);
IBIdiff = abs(RRnC - IBIs2);
    
% Estimation of the resulting ibi time series
correctedIdx = find(IBIdiff > 0.004);
rrTimeSeries.CorrectedPercentage = round(100*length(correctedIdx)/length(RRnC));

%% Show Post-Processing graphics
if GRAPHICS.SHOW_RRS_POST_PROCESSING == true
    figure(4444);
    plot(qrsTimeStamps(2:end)/3600,60.0./RRn, 'k', ...
         qrsTimeStamps(2:end)/3600,60.0./RRnB, 'b', ...
         qrsTimeStamps(2:end)/3600,60.0./RRnC, 'r');
    xlabel('Time from Start [Hour]');
    ylabel('Heart Rate [bpm]');
    title('Ibi Time Series Filtering');
    legend('Error Removal','Adaptive Percent-Filter','Adaptive Controlling Filter');
end

end % End of RRsPostProcessing

