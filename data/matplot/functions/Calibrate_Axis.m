function [  ] = Calibrate_Axis(  )
%   This function defines/re-defines the reference axis

    global X X_d
    global size_x size_y
    global ax bx ay by
    global x_min x_max y_min y_max
    global m_location
    
    fprintf('\t\t\tAxis definition\n');
    
    n = 0;
    A = zeros(4, 4);
    b = zeros(4, 1);

    NULL = 'b';
    set(gcf, 'currentchar', NULL);                              % Initialization

    while(n ~= 4)
        clc
        list = {"Define x reference point", ...
                "Define y reference point", ...
                "Define x/y ratio", ...
                "Read from file", ...
                "Return to the main manu"};
        prompt = {"Calibrate the axis", ...
                  "You need to select two referece x points, and two reference y points", ...
                  sprintf("A total of four reference points are needed (%d / 4)", n), ...
                  "You can load the calibration file", ...
                  "Your last calibration file is automatically saved"};
        i = listdlg('ListString', list, 'SelectionMode', 'single', 'PromptString', prompt, 'ListSize', [400, 300]);

        figure(1);                                  % Focus on the figure
        if (i == 1)                                  % Get x refence point
            while(1)
                c = get(gcf, 'currentchar');        % Get character for control
                Mark_Cursor(0);                     % Mark current mouse cursor to image: cyan
                Display_Image;
                title(sprintf('Coordinate = (%d, %d)\n%s', m_location(1,1), m_location(1,2), ...
                      'Press ''A/a'' to select reference point'), 'fontsize', 12);
                drawnow;
                X_d = X;                            % Initialize image
                                
                if(~length(c)); c = NULL; end;      % Prevent null character - Matlab cannot handle this
                if((c == 'a') || (c == 'A'))        % Select reference point
                    set(gcf, 'currentchar', NULL);  % Reset input
                    figure(1);                      % Focus on the figure
                    
                    [px, py] = m_loc2pixel(m_location(1, 1), m_location(1, 2));
                    rx = (py - bx)/ax;
                    ry = (px - by)/ay;
                    Mark_Cursor(1);                 % Mark current mouse cursor to image: red
                    Display_Image;
                    
                    n = n + 1;                      % Update matrix index
                    alpha = (rx - x_min)/(x_max - x_min);
                    A(n, 1) = 1 - alpha;
                    A(n, 2) = alpha;
                    
                    clc
                    prompt = {'Enter x value of the reference point:'};
                    dlgtitle = 'Reference point: x';
                    data = inputdlg(prompt, dlgtitle);
                    data = cell2mat(data);
                    b(n) = str2double(data);
                    break;                          % Exit while loop
                end
            end
        
        
        elseif (i == 2)                              % Get y refence point
            while(1)
                c = get(gcf, 'currentchar');        % Get character for control
                Mark_Cursor(0);                     % Mark current mouse cursor to image: cyan
                Display_Image;
                title(sprintf('Coordinate = (%d, %d)\n%s', m_location(1,1), m_location(1,2), ...
                      'Press ''A/a'' to select reference point'), 'fontsize', 12);
                drawnow;
                X_d = X;                            % Initialize image
                                
                if(~length(c)); c = NULL; end;      % Prevent null character - Matlab cannot handle this
                if((c == 'a') || (c == 'A'))        % Select reference point
                    set(gcf, 'currentchar', NULL);  % Reset input
                    figure(1);                      % Focus on the figure
                    
                    [px, py] = m_loc2pixel(m_location(1, 1), m_location(1, 2));
                    rx = (py - bx)/ax;
                    ry = (px - by)/ay;
                    Mark_Cursor(1);                 % Mark current mouse cursor to image: red
                    Display_Image;
                    
                    n = n + 1;                      % Update matrix index
                    alpha = (ry - y_min)/(y_max - y_min);
                    A(n, 3) = 1 - alpha;
                    A(n, 4) = alpha;

                    clc;
                    prompt = {'Enter y value of the reference point:'};
                    dlgtitle = 'Reference point: y';
                    data = inputdlg(prompt, dlgtitle);
                    data = cell2mat(data);
                    b(n) = str2double(data);
                    break;                          % Exit while loop
                end
            end
        
            
        elseif (i == 3)                               % Get scale factor
            n = n + 1;
            prompt = {'Ratio for x/y:'};
            dlgtitle = 'Ratio for x/y';
            data = inputdlg(prompt, dlgtitle);
            data = cell2mat(data);
            ratio_xy = str2double(data);
            ratio_xy = ratio_xy * size_x/size_y;
            A(n, 1) = -1;
            A(n, 2) = +1;
            A(n, 3) = +ratio_xy;
            A(n, 4) = -ratio_xy;
            b(n)    = 0;
        

        elseif (i == 4)                               % Get axis data from file
            [Path, ~] = uigetfile( ...
                                  {'*.axis;*.mat', ...
                                   'All axis and matrix files (*.axis,*.mat)'; ...
                                   '*.*', ...
                                   'All Files (*.*)'}, ...
                                   'Select axis calibration file');
            if(isequal(Path,0))
                return;
            end
            data = Read_Matrix(Path);
            x_min = data(1);
            x_max = data(2);
            y_min = data(3);
            y_max = data(4);
            ax = data(5);
            ay = data(6);
            bx = data(7);
            by = data(8);
            Display_Image;
            return;
        

        elseif (isempty(i) || (i == 5))
            figure(1)
            return;

        end
        
    end
    
    if(det(A) == 0)
        fprintf('Failed to calibrate axis.\n');
        pause(1);
        return;
    end
    
    X_d   = X;
    x     = A\b;
    x_min = x(1);
    x_max = x(2);
    y_min = x(3);
    y_max = x(4);
    ax    = (size_x - 1)/(x_max - x_min);
    ay    = (1 - size_y)/(y_max - y_min);
    bx    = size_x - ax*x_max;
    by    = size_y - ay*y_min;
    
    Write_Matrix("./calibration_last.axis", [x_min, x_max, y_min, y_max, ax, ay, bx, by])
    Display_Image;
    

end