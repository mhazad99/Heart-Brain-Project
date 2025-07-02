function qrsPreProcesing = QrsPreProcessingPT(fs, time, ecg_r)
%QRSPREPROCESSING 

global GRAPHICS
global PRE_PROCESSING

%%********** 2- Cancellation of DC Drift ********** %%
ecg_c = ecg_r - mean(ecg_r); % Remove DC conponents

%%********** 3- Bandpass filter for noise cancelation(Filtering)( f1-f2 Hz) ********** %%
Wn=[PRE_PROCESSING.BUTTERWORTH_CUTTOFF_LOW PRE_PROCESSING.BUTTERWORTH_CUTTOFF_HIGH]*2/fs; % cutt off based on fs
[a,b] = butter(PRE_PROCESSING.BUTTERWORTH_ORDER,Wn); % bandpass filtering
try
    ecg_Bpf = filtfilt(a,b,ecg_c);
catch
    ecg_Bpf = ecg_c;
end    

%%********** 4- Normalization of filtered signal ********** %% 
if (max(abs(ecg_Bpf)) > 0)
    ecg_n= ecg_Bpf/max(abs(ecg_Bpf)); 
else
    qrsPreProcesing = [];
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

if GRAPHICS.SHOW_PREPROCESING_STAGES == true
    ShowPreProcessingStages(time, ecg_r, ecg_c, ecg_Bpf, ecg_n, ecgDerive, ecgSquare, ecgInt);
end

qrsPreProcesing.time = time;
qrsPreProcesing.ecg_r = ecg_r;
qrsPreProcesing.ecg_n = ecg_n;
qrsPreProcesing.ecgInt = ecgInt;

end % End of QrsPreProcessingPT

