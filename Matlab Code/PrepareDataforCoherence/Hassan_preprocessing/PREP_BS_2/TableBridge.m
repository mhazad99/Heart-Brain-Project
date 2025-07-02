function [bridgeTS, nbrBridge] = TableBridge(bridgeTS)

if (bridgeTS ~= "")
%convert string to double
    for i =1 :length(bridgeTS.bridgeTS)
        bridgeTS{i,2} = str2double(bridgeTS.bridgeTS{i,2});
    end
    nbrBridge = length(bridgeTS);
end

if (bridgeTS == "")
    nbrBridge = 0;
end

end
