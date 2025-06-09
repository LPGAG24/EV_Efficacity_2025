%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                               Digitizer                                 %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%raaaaa%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath("./functions/"));
i = Initializer;            % Initialize environment
while(i ~= 7)
    i = Get_Input;
    Control(i);
end
clear i
clc;
fprintf('\t\tThe Digitizer Script is terminated\n\n');

global data                 % To store global variable
data = data + 0;            % To get rid of warning