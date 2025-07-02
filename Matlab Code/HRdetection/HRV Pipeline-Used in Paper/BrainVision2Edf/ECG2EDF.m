%% SELECT SOURCE DIRECTORY
%%%%%%%%%%%%%%%%
srcDir = uigetdir(pwd, 'Select Directory with VHDR files');
if srcDir == 0; disp('\tUser has exit'); return; end

%% SELECT DESTINATION DIRECTORY
%%%%%%%%%%%%%%%%
destDir = uigetdir(pwd, 'Select Directory to save EDF files');
if destDir == 0; disp('\tUser has exit'); return; end

allfiles = dir([srcDir filesep '*.vhdr']);
for n=1:length(allfiles)
    loadName=allfiles(n).name;
    dataName=loadName(1:end-5);
    
    % Step2: Import data.
    EEG = pop_loadbv('.', loadName,[],32);
    pop_writeeeg(EEG, [destDir filesep dataName], 'TYPE','EDF');
end