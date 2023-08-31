clc
clear
close all

data = edfread("D:\MastersThesis\OneDrive - ETS\Thesis\Datasets\M-Hassan\Control\RET_1505_RD (23102013).edf"); %Read the dataset
info = edfinfo("D:\MastersThesis\OneDrive - ETS\Thesis\Datasets\M-Hassan\Control\RET_1505_RD (23102013).edf"); %Dataset infromation
fs = info.NumSamples/seconds(info.DataRecordDuration); %samplimg rate


signum = 27; % ECG1 = 15, ECG2 = 16, F3 = 25, M1 = 7, M2 = 8, C3 = 9, O1 = 10
recnum = info.NumDataRecords;
t0 = 0;
timestep = length(data.PULSE{1,1})/fs(signum);
i = -1;
j = -timestep;
TotalY = zeros(recnum*fs(signum)*timestep,1);
for item = 1:recnum
    i = i + 1;
    j = j + timestep;
    t = (t0+(i*timestep):1/fs(signum):(timestep+j-(1/fs(signum))));
    y = data.(signum){item};
    TotalY(length(data.PULSE{1,1})*i+1:item*length(data.PULSE{1,1})) = y;
    %plot(t,y,'b')
    %hold on
end
T = 0:1/fs(signum):(recnum*5)-1/fs(signum);
%Pulscmplx = hilbert(TotalY);
%PulsPhase = angle(Pulscmplx);
%figure;plot(PulsPhase)
figure;plot(T,TotalY,'r')