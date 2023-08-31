function ShowPreProcessingStages(fs, t, ecg_n, ecg_3, ecg_4, ecg_5, ecg_6, ecg_7)
%SHOWPREPROCESSINGSTAGES 

% % Show after centralization and normalization
figure(11)
tN = (0:length(ecg_n)-1)/fs;
plot(tN,ecg_n);
xlabel('Second');
ylabel('Normalized');
title('Centered and Normalized ECG Signal')
grid on;

% Show after LPF 
figure(12)
tLPF = (0:length(ecg_3)-1)/fs;
ecg_3 = ecg_3/max(abs(ecg_3));% normalize , for convenience .
plot(tLPF ,ecg_3);
xlabel('Second');
ylabel('Normalized');
title(' ECG Signal after LPF')
xlim([0 max(tLPF)])
grid on;

% Show after HPF 
figure(13)
tHPF = (0:length(ecg_4)-1)/fs;
ecg_4 = ecg_4/max(abs(ecg_4)); % normalize , for convenience .
plot(tHPF,ecg_4)
xlabel('Second');
ylabel('Normalized');
title(' ECG Signal after HPF')
xlim([0 max(tHPF)])
grid on;

% Show after derivation  
figure(14)
tDer = (0:length(ecg_5)-1)/fs;
ecg_5 = ecg_5/max(abs(ecg_5)); % normalize , for convenience .
plot(tDer,ecg_5)
xlabel('Second');
ylabel('Normalized');
title(' ECG Signal after Derivative')
grid on;

% Show after squaring  
figure(15)
tSq = (0:length(ecg_6)-1)/fs;
ecg_6 = ecg_6/max(abs(ecg_6)); % normalize , for convenience . 
plot(tSq,ecg_6);
xlabel('Second');
ylabel('Normalized');
title(' ECG Signal Squaring')
grid on;

% Show after integration
figure(16)
tInt = (0:length(ecg_7)-1)/fs;
ecg_7 = ecg_7/max(abs(ecg_7)); % normalize , for convenience . 
plot(tN,ecg_n,'b',tInt,ecg_7,'r');
xlabel('Second');
ylabel('Normalized');
title('Integrated ECG Signal')
grid on;

% Show after centralization and normalization
figure(17)
plot(t,ecg_n, ...
    t,ecg_7);
% 	t,ecg_3, ...
% 	t,ecg_4, ...
%	t,ecg_5, ...
%   t,ecg_6, ...
xlabel('Second');
ylabel('Normalized');
title('Pre-Processing Steps')
xlim([t(1) t(end)])
legend('Centralized Input','After Integration');
grid on;

end

