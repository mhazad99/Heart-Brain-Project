function good_label = get_sleep_label(sData)
% input : path, epoch number
% output : list of label for each epoch range

a = {sData.Events.times};
b = numel(a);
b = numel(sData.Events);

if b > 1
    %length for each events
    for i = 1:b
        le(i) = length(a{1, i});
    end
    for i  = 1: length(le)
        if(le(i) > 1)
            g(i) = a{1,i}(2,1) - a{1,i}(1,1);
        else
            g(i) = a{1,1};
        end
    end

    [~, idx_g]= max(g);
    good_label = string(sData.Events(idx_g).label);

end

if b==1
    good_label = string(sData.Events.label);
end

% time_label = string(Comment);

end

