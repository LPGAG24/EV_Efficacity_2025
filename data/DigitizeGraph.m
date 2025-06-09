function DigitizeGraph()
    % Create the main figure
  hFig = figure('Name', 'Graph Digitizer', 'NumberTitle', 'off', 'Position', [100, 100, 1085, 590], 'menubar', 'none', 'Resize', 'off');

    
    % Load image button
    uicontrol('Style', 'pushbutton', 'String', 'LOAD IMAGE', 'Position', [20, 550, 100, 30], 'Callback', @loadImage);


% test field for setting the pixel dimension
  uicontrol('Style', 'text', 'String', 'X-axis pixel dimension', 'HorizontalAlignment', 'left', 'Position', [620, 565, 110, 15], 'Enable', 'off');
  uicontrol('Style', 'edit', 'String', '0.3658537', 'Position', [730 563 100 20], 'TooltipString', 'X-axis pixel dimension', 'Tag', 'Xsize', 'Enable', 'off');
  uicontrol('Style', 'text', 'String', 'Y-axis pixel dimension', 'HorizontalAlignment', 'left', 'Position', [620, 540, 110, 15], 'Enable', 'off');
  uicontrol('Style', 'edit', 'String', '0.0222222', 'Position', [730 538 100 20], 'TooltipString', 'Y-axis pixel dimension', 'Tag', 'Ysize', 'Enable', 'off');

% zero point selection

  uicontrol('Style', 'togglebutton', 'String', 'X0 px', 'Value', 0, 'Position', [858, 563, 40, 21], 'Tag', 'X0_ToggleValue', 'Enable', 'off');
  uicontrol('Style', 'edit', 'String', '0', 'Position', [900 563 60 20], 'TooltipString', 'X0 point selection', 'Tag', 'X0px', 'Enable', 'off');
  uicontrol('Style', 'togglebutton', 'String', 'Y0 px',  'Value', 0, 'Position', [858, 538, 40, 22], 'Tag', 'Y0_ToggleValue', 'Enable', 'off');
  uicontrol('Style', 'edit', 'String', '0', 'Position', [900 539 60 20], 'TooltipString', 'Y0 point selection', 'Tag', 'Y0px', 'Enable', 'off');





    % Clear button
    uicontrol('Style', 'pushbutton', 'String', 'CLEAR', 'Position', [380, 550, 100, 30], 'Callback', @clearData,  'Enable', 'off');
    % Digitize button
    uicontrol('Style', 'pushbutton', 'String', 'DIGITIZE', 'Position', [500, 550, 100, 30], 'Callback', @digitizeGraph,  'Enable', 'off');

    
    % Marker size slider
    uicontrol('Style', 'text', 'String', 'Marker Size', 'Position', [20, 498, 80, 20], 'Enable', 'off');
    hMarkerSize = uicontrol('Style', 'slider', 'Min', 1, 'Max', 500, 'Value', 150, 'Position', [100, 500, 120, 20], 'Callback', @updateMarkerSize,  'Enable', 'off');
    hMarkerSizeValue = uicontrol('Style', 'text', 'String', '', 'HorizontalAlignment', 'left', 'Position', [225, 498, 30, 20], 'Enable', 'off');

    % Threshold factor slider
    uicontrol('Style', 'text', 'String', 'Threshold Factor', 'Position', [270, 498, 100, 20]);
    hThresholdFactor = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.75, 'Position', [370, 500, 120, 20], 'Callback', @updateThresholdFactor,  'Enable', 'off');
    hThresholdFactorValue = uicontrol('Style', 'text', 'String', '75%', 'HorizontalAlignment', 'left',  'Position', [495, 498, 70, 20]);

    % Movmean factor slider
    uicontrol('Style', 'text', 'String', 'Windows size (moving average)', 'Position', [530, 498, 180, 20]);
    hMovFactor = uicontrol('Style', 'slider', 'Min', 1, 'Max', 100, 'Value', 25, 'Position', [700, 500, 120, 20], 'Callback', @updateMovFactor,  'Enable', 'off');
    hMovFactorValue = uicontrol('Style', 'text', 'String', '25', 'HorizontalAlignment', 'left', 'Position', [825, 498, 30, 20]);

    uicontrol('Style', 'checkbox', 'String', ' Detrend', 'value', 1, 'Position', [860 500 100 20], 'Tag', 'Detrend_check', 'Enable', 'off');

    % Image display panel
    hPanel = uipanel('Units', 'pixels', 'Position', [20, 20, 620, 470], 'Title', 'Loaded Image');
    hAx = axes('Parent', hPanel, 'Units', 'normalized', 'Position', [0.05, 0.05, 0.9, 0.9], 'tag', 'ImageDisplay');
    axis(hAx, 'off')
    hImage = [];
    digitizedData = {}; % Use cell array to hold multiple graphs
    userContours = [];
    appendedData = {};
    plotCounter = 0;
    
    % Plot Panel
    hPlotPanel = uipanel('Units', 'pixels', 'Position', [650, 320, 420, 170], 'Title', 'Digitized Graph');
    hPlotAx = axes('Parent', hPlotPanel, 'Units', 'normalized', 'Position', [0.07, 0.1, 0.9, 0.8]);
    axis(hPlotAx, 'off')
    
    % Dropdown menu for graph selection
    hGraphMenu = uicontrol('Style', 'popupmenu', 'String', {'No graphs'}, 'Position', [650, 280, 200, 30], 'Callback', @selectGraph,  'Enable', 'off');

    
    % Add-REPLACE button and counter for adding new graphs
    uicontrol('Style', 'pushbutton', 'String', 'ADD', 'Position', [650, 230, 100, 30], 'Callback', @addGraph,  'Enable', 'off');
    uicontrol('Style', 'pushbutton', 'String', 'REPLACE', 'Position', [755, 230, 100, 30], 'Tag', 'ViewPanel', 'Callback', @replaceGraph,  'Enable', 'off');
    uicontrol('Style', 'pushbutton', 'String', 'DELETE', 'Position', [860, 230, 100, 30], 'Tag', 'ViewPanel', 'Callback', @deleteGraph,  'Enable', 'off');

   hGraphCounter = uicontrol('Style', 'text', 'String', 'Digitized Graphs: 0', 'Position', [870, 287, 200, 20], 'HorizontalAlignment', 'left');

      % Add buttons under the Plot Panel
    uicontrol('Style', 'pushbutton', 'String', 'SAVE EXCEL', 'Position', [650, 190, 100, 30], 'Callback', @saveExcel,  'Tag', 'ViewPanel', 'Enable', 'off');
    uicontrol('Style', 'pushbutton', 'String', 'SAVE TXT', 'Position', [755, 190, 100, 30], 'Callback', @saveTxt,  'Tag', 'ViewPanel', 'Enable', 'off');
    uicontrol('Style', 'pushbutton', 'String', 'VIEW DATA', 'Position', [860, 190, 100, 30], 'Callback', @viewDATA,  'Tag', 'ViewPanel', 'Enable', 'off');
    uicontrol('Style', 'pushbutton', 'String', 'EXPORT TO WS', 'Position', [650, 160, 100, 30], 'Callback', @ExportToWorkspace, 'Tag', 'ViewPanel', 'Enable', 'off');

    % XY point position 
  uicontrol('Style', 'text', 'String', 'х: 0', 'HorizontalAlignment', 'left', 'Position', [25, 40, 100, 13], 'tag', 'Xposition');
  uicontrol('Style', 'text', 'String', 'y: 0', 'HorizontalAlignment', 'left', 'Position', [25, 25, 100, 13], 'tag', 'Yposition');

  set (findobj(gcf, 'Style', 'text'), 'Enable', 'off')


    function updateMarkerSize(~, ~)
        markerSize = get(hMarkerSize, 'Value');
        set(hMarkerSizeValue, 'String', num2str(round(markerSize)));
    end

    function updateThresholdFactor(~, ~)
        thresholdFactor = get(hThresholdFactor, 'Value');
        set(hThresholdFactorValue, 'String', sprintf('%.0f%%', thresholdFactor * 100));
    end

    function updateMovFactor(~, ~)
        MovFactor = get(hMovFactor, 'Value');
        set(hMovFactorValue, 'String', num2str(round(MovFactor)));
    end

function loadImage(~, ~)
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'});
    if isequal(file, 0)
        return;
    end
    if ~isempty(hImage)
        userContours = [];
        img = getappdata(hImage, 'OriginalImageData');
        set(hImage, 'CData', img);
        delete(findall(hAx, 'Tag', 'MarkerArea'));
        delete(findall(hAx, 'Type', 'Line'));
    end
    img = imread(fullfile(path, file));
    if isempty(hImage)
        hImage = imshow(img, 'Parent', hAx);
    else
        set(hImage, 'CData', img);
    end
    setappdata(hImage, 'OriginalImageData', img);
    
    % Stretch image to fill the axes
    set(hImage, 'XData', [0.5 size(img, 2)+0.5], 'YData', [0.5 size(img, 1)+0.5]);
    set(hAx, 'XLim', [0.5 size(img, 2)+0.5], 'YLim', [0.5 size(img, 1)+0.5]);
    set(hAx, 'Units', 'normalized', 'Position', [0 0 1 1]);
    
    

% Calculate new limits based on image width
    imgWidth = size(img, 2);
    imgHeight = size(img, 1);
    newMaxMovFactor = round(0.1 * imgWidth);
    newMaxMarkerSize = round(0.25 * imgWidth);
    newMarkerSizeValue = round(0.075 * imgWidth);
    
    % Update hMovFactor slider
    set(hMovFactor, 'Max', newMaxMovFactor, 'Enable', 'on');
    set(hMovFactorValue, 'String', '25')
    set(hMovFactor, 'Value', 25);
  
    

    % Update hMarkerSize slider
    set(hMarkerSize, 'Max', newMaxMarkerSize, 'Value', newMarkerSizeValue, 'Enable', 'on');
    set(hMarkerSizeValue, 'String', num2str(newMarkerSizeValue));
    


    % Update panel title with file name and image size
    [~, fileName, fileExt] = fileparts(file);
    panelTitle = sprintf('Loaded Image: %s%s %dx%d px', fileName, fileExt, imgWidth, imgHeight);
    set(hPanel, 'Title', panelTitle);
    
    set(hFig, 'WindowButtonDownFcn', @startDrawing);


set(findobj(gcf, 'Style', 'pushbutton'), 'Enable', 'on');
set(hThresholdFactor, 'Enable', 'on');
set(hGraphMenu, 'Enable', 'on');


 set (findobj(gcf, 'Style', 'text'), 'Enable', 'on')
 set (findobj(gcf, 'Style', 'edit'), 'Enable', 'on')
 set (findobj(gcf, 'Style', 'togglebutton'), 'Enable', 'on')
set (findobj(gcf, 'Tag', 'Detrend_check'), 'Enable', 'on')


hViewPanelButtons = findobj(gcf, 'Tag', 'ViewPanel');  
    if isempty(digitizedData) || all(cellfun(@isempty, digitizedData))
         set(hViewPanelButtons, 'Enable', 'off');
    else
        set(hViewPanelButtons, 'Enable', 'on');
    end

 set(hFig, 'WindowButtonMotionFcn', @MarkerLoc);


end


    function startDrawing(~, ~)
        if isempty(hImage)
            return;
        end
        markerSize = get(hMarkerSize, 'Value') / 2;
        
        currPoint = get(hAx, 'CurrentPoint');
        x = currPoint(1, 1);
        y = currPoint(1, 2);

        
         img = getappdata(hImage, 'OriginalImageData');
    imgWidth = size(img, 2);
    imgHeight = size(img, 1);

if (x > 0) && (x < imgWidth) && (y > 0) && (y < imgHeight)
        
  
        hold(hAx, 'on');

        h = patch('XData', x + markerSize./5  * cos(linspace(0, 2*pi, 50)), ...
                  'YData', y + markerSize./5  * sin(linspace(0, 2*pi, 50)), ...
                  'FaceColor', 'black', 'FaceAlpha', 0.07, 'EdgeColor', 'black', 'Tag', 'MarkerArea');

        h = patch('XData', x + markerSize  * cos(linspace(0, 2*pi, 50)), ...
                  'YData', y + markerSize  * sin(linspace(0, 2*pi, 50)), ...
                  'FaceColor', 'yellow', 'FaceAlpha', 0.07, 'EdgeColor', 'none', 'Tag', 'MarkerArea');

        userContours = [userContours; [x, y]];
        set(hFig, 'WindowButtonMotionFcn', @drawMarker);
        set(hFig, 'WindowButtonUpFcn', @stopDrawing);
   
    
      
        % zero point selection and correction
  X0Ch=findobj(gcf, 'Tag', 'X0_ToggleValue');
  Y0Ch=findobj(gcf, 'Tag', 'Y0_ToggleValue');


    if get(X0Ch, 'value')==1
    set (X0Ch, 'Value', 0)
    set (findobj(gcf, 'Tag', 'X0px'), 'string', num2str(fix(x)));
    end
    
    if get(Y0Ch, 'value')==1
    set (Y0Ch, 'Value', 0)
    set (findobj(gcf, 'Tag', 'Y0px'), 'string', num2str(fix(y)));
    end

end
    end

    
    function drawMarker(~, ~)
        
      
        currPoint = get(hAx, 'CurrentPoint');
        x = currPoint(1, 1);
        y = currPoint(1, 2);
        
      
    img = getappdata(hImage, 'OriginalImageData');
    imgWidth = size(img, 2);
    imgHeight = size(img, 1);

if (x > 0) && (x < imgWidth) && (y > 0) && (y < imgHeight)
set(findobj(gcf, 'Tag',  'Xposition'), 'string', ['x: ' num2str(fix(x))]); 
set(findobj(gcf, 'Tag',  'Yposition'), 'string', ['y: ' num2str(fix(y))]);  
 
        
        markerSize = get(hMarkerSize, 'Value') / 2;
        
        
          h = patch('XData', x + markerSize./10  * cos(linspace(0, 2*pi, 50)), ...
                  'YData', y + markerSize./10  * sin(linspace(0, 2*pi, 50)), ...
                  'FaceColor', 'black', 'FaceAlpha', 0.02, 'EdgeColor', 'black', 'EdgeAlpha', 0.05, 'Tag', 'MarkerArea');
        
        patch('XData', x + markerSize * cos(linspace(0, 2*pi, 50)), ...
              'YData', y + markerSize * sin(linspace(0, 2*pi, 50)), ...
              'FaceColor', 'yellow', 'FaceAlpha', 0.02, 'EdgeColor', 'yellow', 'EdgeAlpha', 0.1, 'Tag', 'MarkerArea');
        userContours = [userContours; [x, y]];
end

    end



    function stopDrawing(~, ~)
        set(hFig, 'WindowButtonMotionFcn', @MarkerLoc);
        set(hFig, 'WindowButtonUpFcn', '');
   
    
      

     markerSize = get(hMarkerSize, 'Value') / 2;
        
        currPoint = get(hAx, 'CurrentPoint');
        x = currPoint(1, 1);
        y = currPoint(1, 2);
      
        hold(hAx, 'on');

        h = patch('XData', x + markerSize./5  * cos(linspace(0, 2*pi, 50)), ...
                  'YData', y + markerSize./5  * sin(linspace(0, 2*pi, 50)), ...
                  'FaceColor', 'yellow', 'FaceAlpha', 0.07, 'EdgeColor', 'black', 'Tag', 'MarkerArea');

        h = patch('XData', x + markerSize  * cos(linspace(0, 2*pi, 50)), ...
                  'YData', y + markerSize  * sin(linspace(0, 2*pi, 50)), ...
                  'FaceColor', 'yellow', 'FaceAlpha', 0.07, 'EdgeColor', 'none', 'Tag', 'MarkerArea');

        userContours = [userContours; [x, y]];
        
    end


function MarkerLoc(~, ~)
        currPoint = get(hAx, 'CurrentPoint');
        x = currPoint(1, 1);
        y = currPoint(1, 2);
        
          
 img = getappdata(hImage, 'OriginalImageData');
    imgWidth = size(img, 2);
    imgHeight = size(img, 1);
        if (x > 0) && (x < imgWidth) && (y > 0) && (y < imgHeight)

set(findobj(gcf, 'Tag',  'Xposition'), 'string', ['x: ' num2str(fix(x))]); 
set(findobj(gcf, 'Tag',  'Yposition'), 'string', ['y: ' num2str(fix(y))]);  
        end       

       
end

    function clearData(~, ~)
    if isempty(hImage)
        return;
    end
    userContours = [];
    img = getappdata(hImage, 'OriginalImageData');
    set(hImage, 'CData', img);
    delete(findall(hAx, 'Tag', 'MarkerArea'));
    delete(findall(hAx, 'Type', 'Line'));

    % Do not clear digitizedData and appendedData
end


function digitizeGraph(~, ~)
    if isempty(userContours)
        warndlg('No user contours available.', 'Warning');
        return;
    end
    img = getappdata(hImage, 'OriginalImageData');
    imgWidth = size(img, 2);
    imgHeight = size(img, 1);

    if size(img, 3) == 3
        imgGray = rgb2gray(img);
    else
        imgGray = img;
    end
    delete(findall(hAx, 'Type', 'Line'));
    mask = false(size(imgGray));
    markerSize = get(hMarkerSize, 'Value');
    radius = round(markerSize / 2);
    xCoords = round(userContours(:, 1));
    yCoords = round(userContours(:, 2));
    for i = 1:length(xCoords)
        if xCoords(i) > 0 && xCoords(i) <= size(imgGray, 2) && ...
           yCoords(i) > 0 && yCoords(i) <= size(imgGray, 1)
            mask(yCoords(i), xCoords(i)) = true;
        end
    end
    se = strel('disk', radius, 0);
    mask = imdilate(mask, se);
    mask = imfill(mask, 'holes');
    filteredPoints = [];
    boundaries = bwboundaries(mask);
    if ~isempty(boundaries)
        boundary = boundaries{1};
        thresholdFactor = get(hThresholdFactor, 'Value');
        previousMinY = NaN;
        for x = 1:size(imgGray, 2)
            yValues = find(mask(:, x));
            if ~isempty(yValues)
                intensityValues = double(imgGray(yValues, x));
                intensityThreshold = max(intensityValues) * thresholdFactor;
                validIdx = intensityValues < intensityThreshold;
                validY = yValues(validIdx);
                if ~isempty(validY)
                    if ~isnan(previousMinY)
                        distances = abs(validY - previousMinY);
                        [~, closestIdx] = min(distances);
                        selectedY = validY(closestIdx);
                    else
                        selectedY = median(validY);
                    end
                    previousMinY = selectedY;
                    if inpolygon(x, selectedY, boundary(:, 2), boundary(:, 1))
                        filteredPoints = [filteredPoints; x, selectedY];
                    end
                end
            end
        end
    end
    if ~isempty(filteredPoints)
        MovFactor = get(hMovFactor, 'Value');

        if MovFactor>1
        filteredPoints(:, 2) = movmean(filteredPoints(:, 2), MovFactor);
        end

   


hold(hAx, 'on');
 

filteredPoints2=filteredPoints;

% changing zero position
X0=str2double(get(findobj(gcf, 'Tag', 'X0px'), 'String')); 
Y0=str2double(get(findobj(gcf, 'Tag', 'Y0px'), 'String')); 
filteredPoints(:, 2)=filteredPoints(:, 2)-Y0(1)+imgHeight;
filteredPoints(:, 1)=filteredPoints(:, 1)-X0(1);




% data inversion, taking into account the fact that pixels in the image are counted from top to bottom
filteredPoints(:, 2)=imgHeight-filteredPoints(:, 2);




% Data Interpolation
[sortedX, sortIndex] = sort(filteredPoints(:, 1));
sortedY = filteredPoints(sortIndex, 2);

minX = min(sortedX);
maxX = max(sortedX);
interpolatedX = minX:1:maxX;
interpolatedY = interp1(sortedX, sortedY, interpolatedX, 'spline');
filteredPoints = [interpolatedX', interpolatedY'];






% changing data dimension
Xs=str2double(get(findobj(gcf, 'Tag', 'Xsize'), 'String')); 
Ys=str2double(get(findobj(gcf, 'Tag', 'Ysize'), 'String')); 

filteredPoints(:, 2)=filteredPoints(:, 2).*Ys(1);
filteredPoints(:, 1)=filteredPoints(:, 1).*Xs(1);

DCh=get(findobj(gcf, 'Tag', 'Detrend_check'), 'value');  

% deleting a trend if the corresponding option is selected
if DCh==1
    filteredPoints(:, 2)=detrend(filteredPoints(:, 2));
end

    % Store only the latest digitized graph temporarily, not adding to digitizedData
    setappdata(hFig, 'CurrentGraph', filteredPoints);


        plot(hAx, filteredPoints2(:, 1), filteredPoints2(:, 2), 'r-', 'MarkerSize', 2);
   
    cla(hPlotAx);
        plot(hPlotAx, filteredPoints(:, 1), filteredPoints(:, 2), 'k-');
    end
    title(hPlotAx, 'New Digitized Graph');
    axis off;
end



function addGraph(~, ~)
    currentGraph = getappdata(hFig, 'CurrentGraph');
    if isempty(currentGraph)
        warndlg('No current graph to add.', 'Warning');
        return;
    end

    plotCounter = length(digitizedData) + 1;
    
    % Add new graph
    digitizedData{plotCounter} = currentGraph;

    % Update dropdown menu
    set(hGraphMenu, 'String', arrayfun(@(x) sprintf('Graph %d', x), 1:plotCounter, 'UniformOutput', false));
    
    % Set the dropdown menu to the newly added graph
    set(hGraphMenu, 'Value', plotCounter);
    
    % Update graph counter
    set(hGraphCounter, 'String', sprintf('Number of Digitized Graphs: %d', plotCounter));
  

hViewPanelButtons = findobj(gcf, 'Tag', 'ViewPanel');  
    if isempty(digitizedData) || all(cellfun(@isempty, digitizedData))
         set(hViewPanelButtons, 'Enable', 'off');
    else        
        set(hViewPanelButtons, 'Enable', 'on');
    end

    % Automatically display the new graph
    selectGraph();
end


function replaceGraph(~, ~)
    % Get the current digitized graph
    currentGraph = getappdata(hFig, 'CurrentGraph');
    if isempty(currentGraph)
            warndlg('No current graph to replace.', 'Warning');
        return;
    end
    
    % Get the selected index from the dropdown menu
    selectedIndex = get(hGraphMenu, 'Value');
    
    % Replace the graph at the selected index
    if selectedIndex <= length(digitizedData)
        digitizedData{selectedIndex} = currentGraph;

        
        % Update the plot to reflect the replaced graph
        cla(hPlotAx);
        plot(hPlotAx, currentGraph(:, 1), currentGraph(:, 2), 'k-');
        title(hPlotAx, sprintf('Graph %d', selectedIndex));
        axis off;
    else
        warndlg('Invalid selection index.', 'Warning');
    end

hViewPanelButtons = findobj(gcf, 'Tag', 'ViewPanel');  
    if isempty(digitizedData) || all(cellfun(@isempty, digitizedData))
         set(hViewPanelButtons, 'Enable', 'off');
    else
        set(hViewPanelButtons, 'Enable', 'on');
    end


end


function deleteGraph(~, ~)
    % Get the selected index from the dropdown menu
    selectedIndex = get(hGraphMenu, 'Value');
    
    % Check if the selected index is valid
    if selectedIndex > length(digitizedData) || selectedIndex < 1
             warndlg('Invalid selection index.', 'Warning');
        return;
    end
    
    % Remove the graph at the selected index
    digitizedData(selectedIndex) = [];
     
    % Update the dropdown menu
    numGraphs = length(digitizedData);
    if numGraphs == 0
        set(hGraphMenu, 'String', {'No Graphs'}, 'Value', 1);
        cla(hPlotAx);
        title(hPlotAx, 'No Graphs Available');
        xlabel(hPlotAx, '');
        ylabel(hPlotAx, '');
        axis off;
        set(hPlotAx, 'Visible', 'off');
    else
        set(hGraphMenu, 'String', arrayfun(@(x) sprintf('Graph %d', x), 1:numGraphs, 'UniformOutput', false));
        
        % Adjust the selected index if it was the last graph
        if selectedIndex > numGraphs
            selectedIndex = numGraphs;
        end
        set(hGraphMenu, 'Value', selectedIndex);
        
        % Display the graph at the new selected index
        selectGraph();
    end

   
    % Update graph counter
    set(hGraphCounter, 'String', sprintf('Number of Digitized Graphs: %d', numGraphs));

hViewPanelButtons = findobj(gcf, 'Tag', 'ViewPanel');  
    if isempty(digitizedData) || all(cellfun(@isempty, digitizedData))
         set(hViewPanelButtons, 'Enable', 'off');
    else
        set(hViewPanelButtons, 'Enable', 'on');
    end


end

function selectGraph(~, ~)
    selectedIndex = get(hGraphMenu, 'Value');
    if selectedIndex <= length(digitizedData)
        selectedGraph = digitizedData{selectedIndex};
        cla(hPlotAx);
        if ~isempty(selectedGraph)
            plot(hPlotAx, selectedGraph(:, 1), selectedGraph(:, 2), 'k-');
            title(hPlotAx, sprintf('Graph %d', selectedIndex));
            axis off;
        end
    end
end


  function saveExcel(~, ~)
    if isempty(digitizedData)
           warndlg('No data to save.', 'Warning');
        return;
    end
    [file, path] = uiputfile('*.xlsx', 'Save Excel File');
    if isequal(file, 0)
        return;
    end
    
    % Prepare data for writing
    numGraphs = length(digitizedData);
    maxRows = max(cellfun(@(x) size(x, 1), digitizedData));
    dataMatrix = cell(maxRows + 1, 2 * numGraphs); % Include row for headers
    
    % Generate headers for columns
    for i = 1:numGraphs
        dataMatrix{1, 2*i-1} = sprintf('Graph %d X', i);
        dataMatrix{1, 2*i} = sprintf('Graph %d Y', i);
    end
    
    % Fill data matrix
    for i = 1:numGraphs
        graphData = digitizedData{i};
        rows = size(graphData, 1);
        dataMatrix(2:rows+1, (2*i-1):(2*i)) = num2cell(graphData);
    end
    
    % Write to Excel file
    writecell(dataMatrix, fullfile(path, file));

end






function saveTxt(~, ~)
    if isempty(digitizedData)
       warndlg('No data to save.', 'Warning');
        return;
    end
    [file, path] = uiputfile('*.txt', 'Save TXT File');
    if isequal(file, 0)
        return;
    end
    fileID = fopen(fullfile(path, file), 'w');
    
    numGraphs = length(digitizedData);
    maxRows = max(cellfun(@(x) size(x, 1), digitizedData));
    
    % Write headers
    for i = 1:numGraphs
        fprintf(fileID, 'Graph %d X\tGraph %d Y\t', i, i);
    end
    fprintf(fileID, '\n');
    
    % Write data
    for row = 1:maxRows
        for i = 1:numGraphs
            if row <= size(digitizedData{i}, 1)
                fprintf(fileID, '%.4f\t%.4f\t', digitizedData{i}(row, 1), digitizedData{i}(row, 2));
            else
                fprintf(fileID, '\t\t'); % Empty cells for shorter graphs
            end
        end
        fprintf(fileID, '\n');
    end
    
    fclose(fileID);
end




    function ExportToWorkspace(~, ~)
    if isempty(digitizedData)
        warndlg('No data to export to workspace.', 'Warning');
        return;
    end

    % Copy data to the workspace as a structure
    for i = 1:length(digitizedData)
        varname = sprintf('a%d', i);
        assignin('base', varname, digitizedData{i});
    end

    % Create a structure with all the graphs
    assignin('base', 'digitizedData', digitizedData);
    
    msgbox('Data has been copied to workspace. Access individual graphs as [a1], [a2], etc. or use digitizedData structure.', 'Success');
end

  function viewDATA(~, ~)
    % Check if digitizedData is empty
    if isempty(digitizedData)
        warndlg('No data available to view.', 'Warning');
        return;
    end
    
    % Calculate the number of graphs and the maximum number of rows
    numGraphs = length(digitizedData);
    maxRows = max(cellfun(@(x) size(x, 1), digitizedData));
    
    % Prepare data for uitable
    tableData = cell(maxRows, 2 * numGraphs);
    colNames = cell(1, 2 * numGraphs);
    rowNames = cell(maxRows, 1);  % Add row names for row selection
    
    % Fill data and create column names
    for i = 1:numGraphs
        graphData = digitizedData{i};
        rows = size(graphData, 1);
        tableData(1:rows, (2*i-1):(2*i)) = num2cell(graphData);
        colNames{2*i-1} = sprintf('Graph %d X', i);
        colNames{2*i} = sprintf('Graph %d Y', i);
    end
    
    % Create row names (numbers)
    for i = 1:maxRows
        rowNames{i} = sprintf('%d', i);
    end
    
    % Create a new figure for the table
    hTableFig = uifigure('Name', 'Digitized Graphs Data', 'NumberTitle', 'off', ...
        'Position', [200, 200, 800, 400], 'menubar', 'none');
    
    % Create the table with selection enabled
    t = uitable('Parent', hTableFig, ...
        'Data', tableData, ...
        'ColumnName', colNames, ...
        'RowName', rowNames, ...
        'Units', 'normalized', ...
        'Position', [0 0 1 1], ...
        'tag', 'hTable', ...
        'Enable', 'on', ...
        'ColumnEditable', false, ...
        'RowStriping', 'on', ...
        'SelectionType', 'cell', ... % Enable cell selection
        'MultiSelect', 'on', ... % Enable multiple cell selection
        'CellSelectionCallback', @onCellSelection);
    

    % Create context menu
    cmenu = uicontextmenu(hTableFig);
    t.UIContextMenu = cmenu;
    
    % Add menu items
    uimenu(cmenu, 'Label', 'Copy to Clipboard', 'Callback', @copyToClipboard);
    uimenu(cmenu, 'Label', 'Send to Workspace', 'Callback', @sendToWorkspace);
    
    % Variable to store current selection
    currentSelection = [];
    
    % Callback for cell selection
    function onCellSelection(src, eventdata)
        currentSelection = eventdata.Indices;
    end
    
    % Callback function to copy selected data to clipboard
    function copyToClipboard(~, ~)
        if isempty(currentSelection)
            warndlg('Please select cells first.', 'Warning');
            return;
        end
        
        % Get unique rows and columns from selection
        rows = unique(currentSelection(:,1));
        cols = unique(currentSelection(:,2));
        
        % Extract the selected data
        selectedCells = tableData(rows, cols);
        
        % Convert cells to numeric array, replacing empty and non-numeric cells with NaN
        isNumeric = cellfun(@isnumeric, selectedCells);
        isEmpty = cellfun(@isempty, selectedCells);
        selectedData = nan(size(selectedCells));
        selectedData(isNumeric & ~isEmpty) = cell2mat(selectedCells(isNumeric & ~isEmpty));
        
        % Convert to string and copy to clipboard
        if numel(selectedData) == 1
            if isnan(selectedData)
                clipboard('copy', '');
            else
                % For single value, copy just the number
                clipboard('copy', num2str(selectedData));
            end
        else
            % For multiple values, format as matrix
            dataStr = evalc('disp(selectedData)');
            clipboard('copy', dataStr);
        end
        
        msgbox('Data copied to clipboard!', 'modal');
    end
    
    % Callback function to send selected data to workspace
    function sendToWorkspace(~, ~)
        if isempty(currentSelection)
            warndlg('Please select cells first.', 'Warning');
            return;
        end
        
        % Get unique rows and columns from selection
        rows = unique(currentSelection(:,1));
        cols = unique(currentSelection(:,2));
        
        % Extract the selected data
        selectedCells = tableData(rows, cols);
        
        % Convert cells to numeric array, replacing empty and non-numeric cells with NaN
        isNumeric = cellfun(@isnumeric, selectedCells);
        isEmpty = cellfun(@isempty, selectedCells);
        selectedData = nan(size(selectedCells));
        selectedData(isNumeric & ~isEmpty) = cell2mat(selectedCells(isNumeric & ~isEmpty));
        
        % Create variable name
        varname = 'selected_data';
        base_varname = varname;
        counter = 1;
        
        % Check if variable name exists and increment counter if necessary
        while evalin('base', sprintf('exist(''%s'', ''var'')', varname))
            varname = sprintf('%s%d', base_varname, counter);
            counter = counter + 1;
        end
        
        % Send to workspace
        assignin('base', varname, selectedData);
        msgbox(sprintf('Data sent to workspace as variable "%s"', varname), 'modal');
    end
end


end