function bridged_channels = DetectBridges(eeg, varargin)
% Detect bridged channels based on high correlation for a percentage of the total time OR
%   based on a composite score define using the highest correlation with neighbouring channels
%   during the wole recording
%
% INPUTS:
%   eeg     :   Continuous data set structure, with fields:
%                   - data | [timeSamples, Channels] appropriately high-passed (e.g. >0.5Hz).
%                   - fs    | sampling frequency of data
%
%   varargin    :  possible values order or structure with fields:
%           - 'detect_bridge'   |   detection method: 'Segalowitz_2013' or 'High_Corr'
%           - 'min_corr'    |   minimum correlation to consider for bridging. default 0.99
%           - 'max_dist'    |   Value >= 1 indicate the number of closest neighbors. Values < 1
%                                   indicate a neighbourhood distance. default 0.3
%           - 'window_len'  |   length of the windows (in seconds) for which to compute correlation.
%                                       default 0.5
%           - 'min_corr_time'    |   percentage of the whole signal duration that a correlation
%                                          between neighbouring channels must exceed 'max_corr' to be
%                                           flag as bridged. default 0.8
%
% OUTPUT:
%   bridged_channels : row vector of channel pair indices
%
% Notes:
%   This function requires the Signal Processing toolbox.
% 
% Reference:
%       Desjardins, J. A., & Segalowitz, S. J. (2013). Deconstructing the early visual electrocortical
%       responses to face and house stimuli. Journal of Vision, 13(5):22, 1-18,
%       http://www.journalofvision.org/content/13/5/22, doi:10.1167/13.5.22
% 
% Copyright Tomy Aumont

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1135713 (R2019a) Update 3 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 22-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if number of channels is valid
if size(eeg.data,2) < 2
%     fprintf(2,'PREP> WARNING Can not detect bridges for les than 2 channels\n')
    bridged_channels = [];
    return
end


% Read data structure
if isfield(eeg,'data')
    signal = eeg.data;
else
    fprintf(2,'PREP> ERROR: First argument must be a structure with fields ''data'' and ''fs''\n')
end

if isfield(eeg,'fs')
    fs = eeg.fs;
else
    fprintf(2,'PREP> ERROR: First argument must be a structure with fields ''data'' and ''fs''\n')
end

% Read  dectection criteria
if isstruct(varargin{1})
    detect_method   = varargin{1}.detect_bridges;
    min_corr        = varargin{1}.min_corr;
    max_dist        = varargin{1}.max_dist;
    window_len      = varargin{1}.window_len;
    min_corr_time   = varargin{1}.min_corr_time;
else
    switch length(varargin{1})
        case 1
            detect_method   = varargin{1};
        case 2
            detect_method   = varargin{1};
            min_corr        = varargin{2};
        case 3
            detect_method   = varargin{1};
            min_corr        = varargin{2};
            max_dist        = varargin{3};
        case 4
            detect_method   = varargin{1};
            min_corr        = varargin{2};
            max_dist        = varargin{3};
            window_len      = varargin{4};
        case 5
            detect_method 	= varargin{1};
            min_corr        = varargin{2};
            max_dist        = varargin{3};
            window_len      = varargin{4};
            min_corr_time   = varargin{5};
    end
end

% Assign default value to empty criteria
if ~exist('min_corr','var') || isempty(min_corr);               min_corr = 0.99; end
if ~exist('max_dist','var') || isempty(max_dist);               max_dist = 3; end % 0.03 fit good for hd-EEG-128
if ~exist('window_len','var') || isempty(window_len);           window_len = 0.5; end
if ~exist('min_corr_time','var') || isempty(min_corr_time);     min_corr_time = 0.8; end

% if correlation is too high during this time threshold, flag bridged channels
if min_corr_time > 0 && min_corr_time < 1  %#ok<*NODEF>
    min_corr_time = size(signal,1)*min_corr_time;
else
    min_corr_time = fs*min_corr_time;
end

%% GET NEAREST NEIGHBORS
%%%%%%%%%%%%%%%
neighbours  = GetNeighboursFromPos(eeg.loc, max_dist, eeg.label);
neighIdx    = {neighbours.neighbidx};
% neighIdx = cellfun(@(c) find(c),{neighbours.neighbidx},'UniformOutput',false);


%% DETECT BRIDGES
%%%%%%%%%%%%%%%

% Define windowing parameters
[S,C] = size(signal); % S: number of time points | C: number of channels
window_len = window_len*fs;
wnd = 0:window_len-1;
offsets = 1:window_len:S-window_len+2;
W = length(offsets);

switch lower(detect_method)
    case 'segalowitz_2013'
        % Method from Desjardins & Segalowitz (2013)
        max_corr_r = zeros(C,W); % max correlation matrix per channel per window
        max_corr_ridx = zeros(C,W); % max correlation neighbor index matrix per channel per window
        for o = 1:W
            % Compute correlation between channels
            corr_r = abs(corrcoef(signal(round(offsets(o)+wnd),:)));
            % Keep only THE highest neigbors correlation per channel
            for iChan = 1:C
                if ~isempty(neighIdx{iChan})
                    [max_corr_r(iChan,o),max_corr_ridx(iChan,o)] = max(corr_r(iChan,neighIdx{iChan}));
                end
            end
        end
        % Compute composite score capturing a high and relatively invariant
        %   correlation with neighbors
        mr = mean(max_corr_r,2);
        sr = std(max_corr_r,[],2);
        msr = mr ./ sr;
        % Apply bridging criteria
        bridged_channels = find(msr > 8 * std(msr) + trimmean(msr,25));
        
        if any(bridged_channels)
            %<><><><><><><><><><><>
            % Define bridged electrode PAIRS with the most often maximaly correlated
            %     [e1_1,e1_2; e2_1,e2_2; ...]
            %<><><><><><><><><><><>
            most_freq_neigh = mode(max_corr_ridx,2);
            for iNeigh=1:length(neighIdx)
                neigh(iNeigh,1) = neighIdx{iNeigh}(most_freq_neigh(iNeigh));
            end
            
            bridged_channels = [bridged_channels, neigh(bridged_channels)];
        else
            bridged_channels =[];
        end
    case 'high_corr'
        % Flag channels too correlated with any other channel
        %   (outside the ignored quantile) for each time window
        bridge_flag = zeros(C);
        for o = 1:W
            % Compute correlation between channels
            corr_r = abs(corrcoef(signal(round(offsets(o)+wnd),:)));
            % Flag channel pair with correlation high than threshold
            for iChan = 1:C
                toFlag = corr_r(iChan,neighIdx{iChan}) > min_corr;
                if any(toFlag)
                    bridge_flag(iChan,neighIdx{iChan}(toFlag)) = ...
                        bridge_flag(iChan,neighIdx{iChan}(toFlag)) + 1;
                end
            end
        end
        
        % Ignore redundant pair: ( i, j ) == ( j, i ) AND diagonal
        bridge_flag = tril(bridge_flag,-1);
        % Mark channels for removal if # flagged samples > duration threshold
        [row,col] = find(bridge_flag .* window_len > min_corr_time);
        % Return bridged channels indices
        bridged_channels = [row,col];
        
    otherwise
            disp('PREP>    WARNING: Bridge detection method unrecognized!');
            bridged_channels = [];
end

end

% iFILE392
%  plot(msr, Linewidth = 1.5); hold on;
%  xlabel('electrodes #');
%  ylabel ('msr')
%  yline(8 * std(msr) + trimmean(msr,25));
%  plot(bridged_channels,[msr(52) msr(94)],'.', 'Color', 'k','MarkerSize',50); hold on;
%  legend('' , '', 'PPO2h-PO4' )
% 
% Time = (0:15000-1)/500;   %
% plot(Time, eeg.data(:,31),'k',LineWidth = 1.5); hold on;
% plot(Time, eeg.data(:,63), 'Color', [0 0.4470 0.7410], LineWidth = 1.5);
% xlabel('Temps (s)') ; ylabel('\muV');