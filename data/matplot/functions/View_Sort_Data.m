function [ ] = View_Sort_Data( )
%    This function visualize the data

    global data n
    global x_min x_max y_min y_max
    
    clc;
    data                                    % Data in the form of text
    
    figure(1);
    if(n < 4)
        return;
    end
    method = 'cubic';
    % x_d = Cubic_Spline(1:n, data(:,1), 1:0.1:n);
    % y_d = Cubic_Spline(1:n, data(:,2), 1:0.1:n);
    x_d = interp1(1:n, data(:,1), 1:0.1:n, method);
    y_d = interp1(1:n, data(:,2), 1:0.1:n, method);
    
    while(1)
        Display_Image;
        hold on;
        plot(data(:,1), data(:,2), 'redo', 'linewidth', 2)
        plot(x_d      , y_d      , 'red-', 'linewidth', 2)
        hold off

        list = {"Sort data according to x", ...
                "Sort data according to y", ...
                "Sort data according to the polar angle (concerning the plot's center)", ...
                "Return to the main manu"};
        i = listdlg('ListString', list, 'SelectionMode', 'single', 'ListSize', [400, 300]);

        title(sprintf('Press esc to terminate\n %s', ...
                      'Press ''x/X'' to sort data with respect to x, ''y/Y'' to y'), ...
              'fontsize', 12);
        drawnow;
        
        if (i == 1)
            data = sortrows(data, 1);
            % x_d = Cubic_Spline(1:n, data(:,1), 1:0.1:n);
            % y_d = Cubic_Spline(1:n, data(:,2), 1:0.1:n);
            x_d = interp1(1:n, data(:,1), 1:0.1:n, method);
            y_d = interp1(1:n, data(:,2), 1:0.1:n, method);
            
            clc;
            data
            figure(1);
            
        elseif (i == 2)
            data = sortrows(data, 2);
            % x_d = Cubic_Spline(1:n, data(:,1), 1:0.1:n);
            % y_d = Cubic_Spline(1:n, data(:,2), 1:0.1:n);
            x_d = interp1(1:n, data(:,1), 1:0.1:n, method);
            y_d = interp1(1:n, data(:,2), 1:0.1:n, method);
            
            clc;
            data
            figure(1);

        elseif (i == 3)
            x_bar = (x_min + x_max) / 2;
            y_bar = (y_min + y_max) / 2;
            theta = atan2(data(:, 2) - y_bar, data(:, 1) - x_bar);
            data2 = [theta, data];
            data2 = sortrows(data2, 1);
            data  = data2(:, 2:end);
            clear x_bar y_bar theta data2

            % x_d = Cubic_Spline(1:n, data(:,1), 1:0.1:n);
            % y_d = Cubic_Spline(1:n, data(:,2), 1:0.1:n);
            x_d = interp1(1:n, data(:,1), 1:0.1:n, method);
            y_d = interp1(1:n, data(:,2), 1:0.1:n, method);
            
            clc;
            data
            figure(1);


        elseif (isempty(i) || (i == 4))
            figure(1);                      % Focus on figure again
            return;

        end
        drawnow
    end

end