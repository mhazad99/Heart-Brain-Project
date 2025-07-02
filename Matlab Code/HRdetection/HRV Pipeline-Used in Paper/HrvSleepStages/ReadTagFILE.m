function tagData = ReadTagFILE(stagingFileName)
%ReadTagFILE 
global EPOCH;

    dataString = "";
    fid = fopen(stagingFileName,'r');
  
    %********** Data **********
    tagData.lightsOn_epoch  = NaN;
    tagData.lightsOff_epoch = NaN;
    tagData.lightsOn_time   = "";
    tagData.lightsOff_time  = "";
    tagData.valid           = true;
    
    while(~feof(fid))
        dataString = string(fgetl(fid));
        
        if contains(dataString,'Lights Off','IgnoreCase',true) 
            dummy = string(strsplit(dataString));
            tagData.lightsOff_epoch = str2num(dummy(1));
            if length(dummy) >= 5
                tagData.lightsOff_time = strcat(dummy(4)," ", dummy(5));
            else
                tagData.lightsOff_time = "";
            end    
        end  
        
        if contains(dataString,'Lights On','IgnoreCase',true) 
            dummy = string(strsplit(dataString));
            tagData.lightsOn_epoch = str2num(dummy(1));
            if length(dummy) >= 5
                tagData.lightsOn_time = strcat(dummy(4)," ", dummy(5));
            else
                tagData.lightsOn_time = "";
            end     
        end
    end    
    
  
    fclose(fid);
       
end % End of ReadTagFILE function

