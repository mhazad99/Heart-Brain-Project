function qrsPreProcessing = QrsPreProcessingPT(fs, time, ecg_r, newy, Adjust_num)
%QRSPREPROCESSING 

%%********** 2- Cancellation of DC Drift ********** %%
ecg_c = ecg_r - mean(ecg_r); % Remove DC conponents

%%********** 3- Bandpass filter for noise cancelation(Filtering)( f1-f2 Hz) ********** %%
f1=2;    %%0.5? Rebecca vals:5                                             % cuttoff low frequency to get rid of baseline wander
f2=25;   %%20?  Rebecca vals:15                                            % cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/fs;                                                           % cutt off based on fs
N = 3;                                                                     % order of 3 less processing
[a,b] = butter(N,Wn);                                                      % bandpass filtering
ecg_Bpf = filtfilt(a,b,ecg_c);

%%********** 4- Normalization of filtered signal ********** %% 
if (max(abs(ecg_Bpf)) > 0)
    ecg_n= ecg_Bpf/max(abs(ecg_Bpf)); 
else
    qrsPreProcessing = [];
    return;
end

%********** 5- Derivative Filter ********** %%

% Make impulse response
h = [-1 -2 0 2 1]/8;
ecgDerive = conv(ecg_Bpf,h,'same');

%********** 6- Squaring ********** %%
ecgSquared = ecgDerive.^2;

%********** 7- Moving Window Integration ********** %%

% Make impulse response
h = ones(1 ,31)/31;
ecgInt = conv(ecgSquared,h,'same');
[testpks, testlocs] = findpeaks(ecg_n(1:300*fs), 'MinPeakDistance', 0.5*fs);
meanpeaks = mean(testpks)/5;
[pks,locs] = findpeaks(ecg_n,'MINPEAKDISTANCE',150, 'MinPeakHeight',meanpeaks);
qrsPreProcessing.RRs = diff(time(locs));
HR = 60./qrsPreProcessing.RRs;
%HRcmplx = hilbert(HR);
%qrsPreProcessing.HRPhase = angle(HRcmplx);
newy = newy(Adjust_num + 1:end);
[PHR,fHR] = pwelch(HR,hamming(5120),2560,[],fs);

%if GRAPHICS.SHOW_PREPROCESING_STAGES == true
%ShowPreProcessingStages(time, ecg_r, ecg_c, ecg_Bpf, ecg_n, ecgDerive, ecgSquared, ecgInt);
%end

qrsPreProcessing.time = time;
qrsPreProcessing.ecg_r = ecg_r;
qrsPreProcessing.ecg_n = ecg_n;
qrsPreProcessing.ecgInt = ecgInt;
figure
subplot(4,2,1);plot(time, ecg_r);title('Raw signal');subplot(4,2,2);plot(time, ecg_c);title('DC Cancelation');
subplot(4,2,3);plot(time, ecg_Bpf);title('Bandpass filtered');subplot(4,2,4);plot(time, ecg_n);title('Normalized');hold on;plot(time(locs), pks, 'r*')
subplot(4,2,5);plot(time, ecgDerive);title('Derivative');subplot(4,2,6);plot(time, ecgSquared);title('Squared');
subplot(4,2,7);plot(time, ecgInt, 'b-');title('Integration');
figure;plot(HR);title('HR with peak detection')
%figure;plot(qrsPreProcessing.HRPhase);title('Phase of HR')
figure;plot(fHR, PHR);title('PSD of HR peak detection')
end % End of QrsPreProcessingPT

