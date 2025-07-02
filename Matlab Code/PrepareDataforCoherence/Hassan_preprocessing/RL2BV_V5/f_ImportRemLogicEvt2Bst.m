function f_ImportRemLogicEvt2Bst(RawFile,evt)
% Fill raw files with sleep scoring events
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Right-click on a "Link to raw file" in the database explorer
%  > File > Copy file path to clipboard
% RawFile = {'/media/taumont/SAM/SAM/data/SAM_bst_db/SAM001/data/SAM001n1/@rawSB001_0330_2305/data_0raw_SB001_0330_2305.mat'};%,...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM001/data/SAM001n2/@rawSB001_0406_2304/data_0raw_SB001_0406_2304.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM002/data/SAM002n1/@rawSAM-002_0510_2329/data_0raw_SAM-002 0510 2329.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM002/data/SAM002n2/@rawSAM-002_0519_2250/data_0raw_SAM-002_0519_2250.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM003/data/SAM003n1/@rawSAM-003_0414_2326/data_0raw_SAM-003_0414_2326.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM003/data/SAM003n2/@rawSAM-003_0421_2324/data_0raw_SAM-003 0421 2324.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM005/data/SAM005n1/@rawSAM-005_0624_2259/data_0raw_SAM-005_0624_2259.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM005/data/SAM005n2/@rawSAM-005_0707_2313/data_0raw_SAM-005_0707_2313.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM008/data/SAM008n1/@rawSAM008_0714_2322/data_0raw_SAM008_0714_2322.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM008/data/SAM008n2/@rawSAM008_0725_2328/data_0raw_SAM008_0725_2328.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM010/data/SAM010n1/@rawSAM010_0726_2346/data_0raw_SAM010_0726_2346.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM010/data/SAM010n2/@rawSAM010_0809_2352/data_0raw_SAM010_0809_2352.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM014/data/SAM014n1/@rawSAM-014_0927_2337/data_0raw_SAM-014_0927_2337.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM015/data/SAM015n1/@rawSAM-015_1006_2257/data_0raw_SAM-015_1006_2257.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM015/data/SAM015n2/@rawSAM-015_1013_2258/data_0raw_SAM-015_1013_2258.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM017/data/SAM017n1/@rawSAM-017_1027_2344/data_0raw_SAM-017_1027_2344.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM017/data/SAM017n2/@rawSAM-017_1111_0002/data_0raw_SAM-017_1111_0002.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM019/data/SAM019n1/@rawSAM01925JUN1935_1108_2110/data_0raw_SAM01925JUN1935_1108_2110.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM020/data/SAM020n1/@rawSAM-020_1124_2301/data_0raw_SAM-020_1124_2301.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM023/data/SAM023n1/@raw28_JUN_1945_0413_2317/data_0raw_28_JUN_1945_0413_2317.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM023/data/SAM023n2/@raw28_JUN_1945_0427_2324/data_0raw_28_JUN_1945_0427_2324.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM026/data/SAM026n2/@rawSAM-026_0608_2259/data_0raw_SAM-026_0608_2259.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM028/data/SAM028n1/@rawSAM-028_0706_2214/data_0raw_SAM-028_0706_2214.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM028/data/SAM028n2/@rawSAM-028_0713_2231/data_0raw_SAM-028_0713_2231.mat',...
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM030/data/SAM030n1/@rawSAM-030_0726_2253/data_0raw_SAM-030_0726_2253.mat',... 30n1 part 1
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM030/data/SAM030n1/@rawSAM-030_0726_2319/data_0raw_SAM-030_0726_2319.mat',... 30n1 part 2
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM030/data/SAM030n2/@rawSAM-030_0810_2325/data_0raw_SAM-030_0810_2325.mat',... 30n2 part 1
%     '/media/taumont/SAM/SAM/data/SAM_bst_db/SAM030/data/SAM030n2/@rawSAM-030_0811_0307/data_0raw_SAM-030_0811_0307.mat'}; % 30n2 part 2

ColorTable = ...
    [0     1    0   
    .4    .4    1
     1    .6    0
     0     1    1
    .56   .01  .91
     0    .5    0
    .4     0    0
     1     0    1
    .02   .02   1
    .5    .5   .5];


tCol   = contains(evt(1,:),'Time');         % Column index of event time
eCol   = contains(evt(1,:),'Event');        % Column index of event name
dCol   = contains(evt(1,:),'Duration');     % Column index of event duration
epCol  = contains(evt(1,:),'Epoch');        % Column index of epoch number
slpCol = contains(evt(1,:),'Sleep Stage');  % Column index of sleep stages



% for iFile = 1:length(RawFile)
    % Load the "sFile" structure, contained in the .F structure of the link file (data_0raw...mat)
    fprintf('\t\tReading Brainstorm raw file... it may take few minutes\n'); tic
    sRaw = load(RawFile, 'F');
    fprintf('\t\t Loading time: %f seconds\n', toc);
    
    fprintf('\t\tImporting events...\n')
    recStart = sRaw.F.header.EEG.etc.recordingtime; % datenum format
%     fs = sRaw.F.prop.sfreq;
    % ===== GET EVENT TYPES =====
    if any(slpCol)
%         evtName = evt{iEvt,slpCol};
        evtTypes = unique(evt(2:end,slpCol));
        evtCol = slpCol;
        if isempty(evtTypes) || (length(evtTypes)==1 && strcmpi(evtTypes,'N/A'))
           if any(eCol)
               evtTypes = unique(evt(2:end,eCol));
               evtCol = eCol;
           end
        end
    elseif any(eCol)
        evtTypes = unique(evt(2:end,eCol));
        evtCol = eCol;
    end
    
    for iType = 1:length(evtTypes)
        evtTypeIdx = find(~cellfun(@(c)~contains(c,evtTypes{iType}),{sRaw.F.events.label}));
        if isempty(evtTypeIdx)
            % Get new event category indice
            evtTypeIdx = length(sRaw.F.events) + 1;
            % Name the new event
            sRaw.F.events(evtTypeIdx).label = evtTypes{iType};
            isNewEvent = true;
        else
            isNewEvent = false;
        end
        
        % Get all event from that category and add them to the file
        evtList = find(strcmpi(evt(:,evtCol),evtTypes{iType}));
        for iEvt = 1:length(evtList)
            % Event onset in smaples
            evtOnsetPos = f_GetEvtPos(recStart,evt{evtList(iEvt),tCol});
            % Event offset in samples
            evtDuration = str2double(evt{evtList(iEvt),dCol});
            evtOffsetPos = evtOnsetPos + round(evtDuration);
            
            if ~isNewEvent
                if any(sum(ismember(sRaw.F.events(evtTypeIdx).times,[evtOnsetPos;evtOffsetPos]))==2)
                    continue;
                end
            end
            % Apply offset to the events in the "button_offset" category
            sRaw.F.events(evtTypeIdx).times = [sRaw.F.events(evtTypeIdx).times [evtOnsetPos;evtOffsetPos]];
            % Round new time values to the nearest sample
            sRaw.F.events(evtTypeIdx).times = ...
                round(sRaw.F.events(evtTypeIdx).times .* sRaw.F.prop.sfreq) ./ sRaw.F.prop.sfreq;
            % Re-generate an epochs field with only ones, and empty notes and channels fields
            % (optional here, as we didn't change the number of evt)
            nTimes = size(sRaw.F.events(evtTypeIdx).times, 2);
            sRaw.F.events(evtTypeIdx).epochs = ones(1, nTimes);
            sRaw.F.events(evtTypeIdx).channels = cell(1, nTimes);
            sRaw.F.events(evtTypeIdx).notes = cell(1, nTimes);
            % Change the event color to yellow (red=1, green=1, blue=0)
            if iType > size(ColorTable,1)
                if iColor < size(ColorTable,1)
                    iColor = iColor + 1;
                else
                    iColor = 1;
                end
            else
                iColor = iType;
            end
            sRaw.F.events(evtTypeIdx).color = ColorTable(iColor,:);
        end
    end
    
    % Update the sRaw structure to the RawFile file
    fprintf('\t\tSaving Brainstorm raw file...\n'); tic
    save(RawFile, '-struct', 'sRaw', '-v7.3');
    fprintf('\t\t Saving time: %f seconds\n', toc);
%     bst_save(RawFile, sRaw, 'v7.3', 1);
end


