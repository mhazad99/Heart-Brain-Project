function textHeader = GetHeader()
% GetHeader:
%   Returns the Text Header to add at the start of the output text file.

textHeader = string.empty;
patientName = "Patient Name: ";
projectName = "Project: ";
hospitalNumber = "Hospital #: ";
subjectCode = "Subject Code: ";
studyDate = "Study Date: ";
ssc_sin = "SSC/SIN:	";
gender = "Sex: ";
dob = "D.O.B: ";
age = "Age:	";
height = "Height: ";
weight = "Weight: ";
bmi = "B.M.I: ";

epoch = "Epoch";
event = "Event";

textHeader = sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n%s %s\n", ...
    patientName, projectName, hospitalNumber, subjectCode, studyDate, studyDate, ...
    ssc_sin, gender, dob, age, height, weight, bmi, epoch, event);

end % End of GetHeader function