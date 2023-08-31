function qrsDetection = QrsDetectionPT(fs, qrsPreProcesing, newy, Adjust_num)
%QRSDETECTIONPT
%
%
%%**********  Initialize**********
delay = 0;
skip = 0; % becomes one when a T wave is detected
m_selected_RR = 0;
mean_RR = 0;
searh_back = 0;
ax = zeros(1,6);

%%********** Raw EKG signal **********
ecg_r =  qrsPreProcesing.ecg_r;

%%********** Normalized and filtered signal **********
ecg_n = qrsPreProcesing.ecg_n;

%%********** Integrated average signal **********
ecg_int = qrsPreProcesing.ecgInt;
T_WAVE_LOCATION = 0.360;
NB_T_WAVE_LOCATION_SAMPLES = round(T_WAVE_LOCATION*fs);
PEAK_SEARCH_DURATION = 0.150;
NB_PEAK_SEARCH_SAMPLES = round(PEAK_SEARCH_DURATION*fs);
SLOPE_SEARCH_DURATION = 0.075;
NB_SLOPE_SEARCH_SAMPLES = round(SLOPE_SEARCH_DURATION*fs);

%%********** Fiducial Marks **********%%
% Note : a minimum distance of 40 samples is considered between each R wave
% since in physiological point of view no RR wave can occur in less than
% 200 msec distance.
REFACTORY_PERIOD = 0.200;
NB_REFACTORY_SAMPLES = round(REFACTORY_PERIOD*fs);
try
    [pks,locs] = findpeaks(ecg_int,'MINPEAKDISTANCE',NB_REFACTORY_SAMPLES);
catch
    pks = [];
    locs = [];
end
%%********** Initialize of Parameters **********%
nbPeaks = length(pks);

%********** Stores QRS wrt Sig and Filtered Sig **********%
qrs_c = zeros(1,nbPeaks);           % amplitude of R
qrs_i = zeros(1,nbPeaks);           % index
qrs_i_raw = zeros(1,nbPeaks);       % amplitude of R
qrs_amp_raw= zeros(1,nbPeaks);      % Index

%********** Noise Buffers **********%
nois_c = zeros(1,nbPeaks);
nois_i = zeros(1,nbPeaks);

%********** Buffers for Signal and Noise **********%
SPKI_buf = zeros(1,nbPeaks);
NPKI_buf = zeros(1,nbPeaks);
SPKF_buf = zeros(1,nbPeaks);
NPKF_buf = zeros(1,nbPeaks);
THF1_buf = zeros(1,nbPeaks);
THI1_buf = zeros(1,nbPeaks);

%********** Initialize the training phase (2 seconds of the signal) to **********%
LEARNING1_DURATION = 2.0;
NB_LEARNING1_SAMPLES = min(LEARNING1_DURATION*fs,length(ecg_int));

%********** Initialize the integrated signal thresholds (THR_SIG and THR_NOISE) **********%
THI1 = max(ecg_int(1:NB_LEARNING1_SAMPLES))*1/3; % 0.333 of the max amplitude
THI2 = mean(ecg_int(1:NB_LEARNING1_SAMPLES))*1/2;% 0.5 of the mean signal is considered to be noise
SPKI= THI1; % Signal level in integrated signal
NPKI = THI2; % Noise level in integrated signal

%********** Initialize the bandpass signal thresholds (THI11 and THI21) **********%
THF1 = max(ecg_n(1:NB_LEARNING1_SAMPLES))*1/3; % 0.333 of the max amplitude
THF2 = mean(ecg_n(1:NB_LEARNING1_SAMPLES))*1/2; % 0.5 of the mean signal is considered to be noise
SPKF = THF1;  % Signal level in Bandpassed signal
NPKF = THF2; % Noise level in Bandpassed signal

%%********** Thresholding and desicion rule **********%%
Beat_C = 0;      % Raw Beats
Beat_C1 = 0;     % Filtered Beats
Noise_Count = 0; % Noise Counter

for i = 1 : nbPeaks
    %%********** Locate the corresponding peak in the filtered signal ********** %%
    if locs(i)- NB_PEAK_SEARCH_SAMPLES >= 1 && locs(i) <= length(ecg_n)
        [y_i,x_i] = max(ecg_n(locs(i)- NB_PEAK_SEARCH_SAMPLES:locs(i)));
    else
        if i == 1
            [y_i,x_i] = max(ecg_n(1:locs(i)));
            searh_back = 1;
        elseif locs(i)>= length(ecg_n)
            [y_i,x_i] = max(ecg_n(locs(i)- NB_PEAK_SEARCH_SAMPLES:end));
        end
    end

    %%********** Update the heart_rate **********%%
    if Beat_C >= 9
        diffRR = diff(qrs_i(Beat_C-8:Beat_C)); % calculate RR interval
        mean_RR = mean(diffRR); % calculate the mean of 8 previous R waves interval
        comp =qrs_i(Beat_C)-qrs_i(Beat_C-1); % latest RR

        if comp <= 0.92*mean_RR || comp >= 1.16*mean_RR
            %********** lower down thresholds to detect better in MVI **********%
            THI1 = 0.5*(THI1);
            THF1 = 0.5*(THF1);
        else
            m_selected_RR = mean_RR;  % The latest regular beats mean
        end

    end

    %%********** calculate the mean last 8 R waves **********%%
    if m_selected_RR
        test_m = m_selected_RR; %if regular RR availabe use it
    elseif mean_RR && m_selected_RR == 0
        test_m = mean_RR;
    else
        test_m = 0;
    end

    if test_m
        if (locs(i) - qrs_i(Beat_C)) >= round(1.66*test_m)  % Missed QRS
            % Search back and locate the max in this interval
            startIdx = qrs_i(Beat_C)+ NB_REFACTORY_SAMPLES;
            endIdx = locs(i)- NB_REFACTORY_SAMPLES;
            [pks_temp,locs_temp] = max(ecg_int(startIdx:endIdx));
            locs_temp = qrs_i(Beat_C)+ NB_REFACTORY_SAMPLES + locs_temp -1;% location

            if pks_temp > THI2
                Beat_C = Beat_C + 1;
                qrs_c(Beat_C) = pks_temp;
                qrs_i(Beat_C) = locs_temp;

                %********** Locate in Filtered Sig **********%
                if locs_temp <= length(ecg_n)
                    [y_i_t,x_i_t] = max(ecg_n(locs_temp- NB_PEAK_SEARCH_SAMPLES:locs_temp));
                else
                    [y_i_t,x_i_t] = max(ecg_n(locs_temp- NB_PEAK_SEARCH_SAMPLES:end));
                end

                %********** Band pass Sig Threshold **********%
                if y_i_t > THF2
                    Beat_C1 = Beat_C1 + 1;
                    qrs_i_raw(Beat_C1) = locs_temp- NB_PEAK_SEARCH_SAMPLES + ...
                        (x_i_t - 1);% save index of bandpass
                    qrs_amp_raw(Beat_C1) = y_i_t; % save amplitude of bandpass
                    SPKF = 0.25*y_i_t + 0.75*SPKF; % when found with the second thres
                end

                SPKI = 0.25*pks_temp + 0.75*SPKI ; % when found with the second threshold
            end
        end
    end

    %%********** find noise and QRS peaks **********%%
    if pks(i) >= THI1

        %********** if No QRS in 360ms of the previous QRS See if T wave **********%
        if Beat_C >= 3 % After learning phase #2
            if (locs(i)-qrs_i(Beat_C)) <= NB_T_WAVE_LOCATION_SAMPLES
                % mean slope of the waveform at that position
                Slope1 = mean(diff(ecg_int(locs(i)- NB_SLOPE_SEARCH_SAMPLES:locs(i))));
                % mean slope of previous R wave
                Slope2 = mean(diff(ecg_int(qrs_i(Beat_C)- NB_SLOPE_SEARCH_SAMPLES:qrs_i(Beat_C))));

                if abs(Slope1) <= abs(0.5*(Slope2)) % slope less then 0.5 of previous R
                    Noise_Count = Noise_Count + 1;
                    nois_c(Noise_Count) = pks(i);
                    nois_i(Noise_Count) = locs(i);
                    skip = true;  % T wave identification
                    % ----- adjust noise levels ------ %
                    NPKF = 0.125*y_i + 0.875*NPKF;
                    NPKI = 0.125*pks(i) + 0.875*NPKI;
                else
                    skip = false;
                end
            end
        end

        %********** skip is 1 when a T wave is detected **********%
        if skip == false
            Beat_C = Beat_C + 1;
            qrs_c(Beat_C) = pks(i);
            qrs_i(Beat_C) = locs(i);

            %**********bandpass filter check threshold **********%
            if y_i >= THF1
                Beat_C1 = Beat_C1 + 1;
                if searh_back
                    qrs_i_raw(Beat_C1) = x_i; % save index of bandpass
                else
                    qrs_i_raw(Beat_C1)= locs(i)-round(0.150*fs)+(x_i - 1); % save index of bandpass
                end
                qrs_amp_raw(Beat_C1) =  y_i; % save amplitude of bandpass
                SPKF = 0.125*y_i + 0.875*SPKF; % adjust threshold for bandpass filtered sig
            end
            SPKI = 0.125*pks(i) + 0.875*SPKI ; % adjust Signal level
        end

    elseif (THI2 <= pks(i)) && (pks(i) < THI1)

        NPKF = 0.125*y_i + 0.875*NPKF;  % adjust Noise level in filtered sig
        NPKI = 0.125*pks(i) + 0.875*NPKI; % adjust Noise level in MVI

    elseif pks(i) < THI2

        Noise_Count = Noise_Count + 1;
        nois_c(Noise_Count) = pks(i);
        nois_i(Noise_Count) = locs(i);
        NPKF = 0.125*y_i + 0.875*NPKF; % noise level in filtered signal
        NPKI = 0.125*pks(i) + 0.875*NPKI; % adjust Noise level in MVI

    end

    %%********** adjust the threshold with SNR **********%%
    if NPKI ~= 0 || SPKI ~= 0
        THI1 = NPKI + 0.25*(abs(SPKI - NPKI));
        THI2 = 0.5*(THI1);
    end

    %********** adjust the threshold with SNR for bandpassed signal **********%
    if NPKF ~= 0 || SPKF ~= 0
        THF1 = NPKF + 0.25*(abs(SPKF - NPKF));
        THF2 = 0.5*(THF1);
    end

    %********** take a track of thresholds of smoothed signal **********%
    SPKI_buf(i) = SPKI;
    NPKI_buf(i) = NPKI;
    THI1_buf(i) = THI1;

    %********** take a track of thresholds of filtered signal **********%
    SPKF_buf(i) = SPKF;
    NPKF_buf(i) = NPKF;
    THF1_buf(i) = THF1;

    %********** reset parameters **********%
    skip = 0;
    searh_back = 0;
end

%%********** Adjust Lengths **********%%
qrs_i_raw = qrs_i_raw(1:Beat_C1);
qrs_amp_raw = qrs_amp_raw(1:Beat_C1);
qrs_c = qrs_c(1:Beat_C);
qrs_i = qrs_i(1:Beat_C);

Ts = 1.0/fs;
AMP_LOWER_THRESHOLD = 1.0e-6;
noiseIdx = find(qrs_amp_raw <= AMP_LOWER_THRESHOLD);
qrs_amp_raw(noiseIdx) = [];
qrs_i_raw(noiseIdx) = [];

qrsDetection.qrs_amp = qrs_amp_raw;
qrsDetection.qrs_i_raw = qrs_i_raw;
qrsDetection.qrs_time = qrs_i_raw.*Ts;

qrsDetection.rr_intervals = diff(qrsDetection.qrs_time);

%if GRAPHICS.SHOW_QRS_DETECTION == true
figure(111);
az(1)=subplot(311);
plot(ecg_n);
title('QRS on Filtered Signal');
axis tight;
hold on,scatter(qrs_i_raw,qrs_amp_raw,'m');
hold on,plot(locs,NPKF_buf,'LineWidth',2,'Linestyle','--','color','k');
hold on,plot(locs,SPKF_buf,'LineWidth',2,'Linestyle','-.','color','r');
hold on,plot(locs,THF1_buf,'LineWidth',2,'Linestyle','-.','color','g');
hold off

az(2)=subplot(312);plot(ecg_int);
title(sprintf('QRS on MVI signal, Noise level(black)\nSignal Level (red) and Adaptive Threshold(green)'));
axis tight;
hold on,scatter(qrs_i,qrs_c,'m');
hold on,plot(locs,NPKI_buf,'LineWidth',2,'Linestyle','--','color','k');
hold on,plot(locs,SPKI_buf,'LineWidth',2,'Linestyle','-.','color','r');
hold on,plot(locs,THI1_buf,'LineWidth',2,'Linestyle','-.','color','g');
hold off


az(3)=subplot(313);
plot(ecg_n);
title('Pulse train of the found QRS on ECG signal');
axis tight;
line(repmat(qrs_i_raw,[2 1]),...
    repmat([min(ecg_n)/2; max(ecg_n)/2],size(qrs_i_raw)),...
    'LineWidth',2.5,'LineStyle','-.','Color','r');
linkaxes(az,'x');
zoom on;
pause(1)

figure
HR = 60./qrsDetection.rr_intervals;
plot(HR);title('HR with complete PT algo')
figure
newy = newy(Adjust_num + 1:end);
[PHR,fHR] = pwelch(HR,hamming(5120),2560,[],fs);

plot(fHR, PHR);title('PSD of HR with complete PT algo')
%end

end % End of QrsDetectionPT function




