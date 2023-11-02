function [data] = check_packet_loss(log_name)
    fileID = fopen(log_name, 'r');
    contents = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    % read all the lines
    logLines = contents{1};
    data = [];
    for i = 1:length(logLines)
        line = logLines{i};
        % only extract second here, since in the same minute, o.w. (\d+:\d+\.\d+)
        tokens = regexp(line, '(\d+\.\d+).*INFO - (\d+)', 'tokens');
        if isempty(tokens)
            tokens = regexp(line, '(\d+\.\d+).*INFO - data_from_simulink (\d+)', 'tokens');
        end
        if ~isempty(tokens)
            time_stamp = str2double(tokens{1}{1});  % Extract the time stamp
            data_value = str2double(tokens{1}{2});  % Extract and convert data to a numeric value
            % Store the extracted data (time stamp and data) as needed
            data = [data; time_stamp, data_value];
        else
            fprintf('empty field here at %d line\n', i);
        end
    end
    % normalize time
    timeVec = data(:, 1) - data(1, 1);
    % packet received
    packetRecv = data(:, 2)/12;
    % packet loss
    
    % Plot the time series
    figure()
    plot(timeVec, packetRecv);
    hold on
    figure()
    plot(timeVec, packetRecv);
end