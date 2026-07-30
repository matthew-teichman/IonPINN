% plot_transients.m
% Script to plot 5 transients from Testing, Training, and Validation datasets

base_dir = fullfile('..', 'data');
plot_out_dir = 'plot';
if ~exist(plot_out_dir, 'dir')
    mkdir(plot_out_dir);
end

datasets = {'Testing', 'Training', 'Validation'};

for i = 1:length(datasets)
    dataset_name = datasets{i};
    dataset_dir = fullfile(base_dir, dataset_name);
    
    if exist(dataset_dir, 'dir')
        % Get all CSV files in the directory
        csv_files = dir(fullfile(dataset_dir, '*.csv'));
        
        % Filter out very small files that might be invalid
        valid_files = {};
        for j = 1:length(csv_files)
            if csv_files(j).bytes > 1000
                valid_files{end+1} = csv_files(j);
            end
        end
        
        % Plot up to 5 valid transients
        num_files_to_plot = min(5, length(valid_files));
        
        if num_files_to_plot > 0
            figure('Name', sprintf('%s Transients', dataset_name), 'NumberTitle', 'off', 'Position', [100, 100, 800, 900]);
            
            for k = 1:num_files_to_plot
                filepath = fullfile(valid_files{k}.folder, valid_files{k}.name);
                
                % Try to read the CSV file
                try
                    opts = detectImportOptions(filepath);
                    data = readtable(filepath, opts);
                    
                    % Check if expected columns are present
                    hasVars = ismember({'Time', 'Voltage', 'Current', 'Temperature'}, data.Properties.VariableNames);
                    if all(hasVars)
                        [~, name, ~] = fileparts(valid_files{k}.name);
                        
                        % Voltage Subplot
                        subplot(3, 1, 1);
                        hold on;
                        plot(data.Time, data.Voltage, 'DisplayName', name, 'LineWidth', 1.5);
                        
                        % Current Subplot
                        subplot(3, 1, 2);
                        hold on;
                        plot(data.Time, data.Current, 'DisplayName', name, 'LineWidth', 1.5);
                        
                        % Temperature Subplot
                        subplot(3, 1, 3);
                        hold on;
                        plot(data.Time, data.Temperature, 'DisplayName', name, 'LineWidth', 1.5);
                    else
                        warning('File %s is missing required columns (Time, Voltage, Current, or Temperature)', valid_files{k}.name);
                    end
                catch ME
                    warning('Failed to read %s: %s', valid_files{k}.name, ME.message);
                end
            end
            
            % Format Voltage Subplot
            subplot(3, 1, 1);
            title(sprintf('Voltage Transients - %s Data', dataset_name));
            ylabel('Voltage (V)');
            legend('show', 'Location', 'best');
            grid on;
            hold off;
            
            % Format Current Subplot
            subplot(3, 1, 2);
            title(sprintf('Current Transients - %s Data', dataset_name));
            ylabel('Current (A)');
            grid on;
            hold off;
            
            % Format Temperature Subplot
            subplot(3, 1, 3);
            title(sprintf('Temperature Transients - %s Data', dataset_name));
            xlabel('Time (s)');
            ylabel('Temperature (C)');
            grid on;
            hold off;
            
            % Save the plot as a PNG image in the plot folder
            out_filename = fullfile(plot_out_dir, sprintf('%s_transients.png', lower(dataset_name)));
            saveas(gcf, out_filename);
            fprintf('Saved %s\n', out_filename);
        else
            fprintf('No valid CSV files found in %s\n', dataset_dir);
        end
    else
        fprintf('Directory %s does not exist.\n', dataset_dir);
    end
end
