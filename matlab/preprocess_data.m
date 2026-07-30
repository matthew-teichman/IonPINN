% Preprocessing Script for IonPINN - Phase 1
% Reads NASA Battery Datasets (Dataset 5 and 11), applies zero-phase filtering, 
% and exports flat .csv files cycle-by-cycle.

clear; close all; clc;

% Setup Paths
base_dir = fullfile(pwd, '..'); % Assuming we run from the matlab/ directory
datasets_dir = fullfile(base_dir, 'datasets');
data_out_dir = fullfile(base_dir, 'data');

% Ensure output directories exist
folders = {'Training', 'Validation', 'Testing', 'Unused'};
for i = 1:length(folders)
    out_path = fullfile(data_out_dir, folders{i});
    if ~exist(out_path, 'dir')
        mkdir(out_path);
    end
end

% Define file mappings
% Struct format: category, filename, relative_path
files_to_process = {
    % Training
    'Training', 'B0005.mat', '5. Battery Data Set/1. BatteryAgingARC-FY08Q4';
    'Training', 'B0006.mat', '5. Battery Data Set/1. BatteryAgingARC-FY08Q4';
    'Training', 'RW1.mat', '11. Randomized Battery Usage Data Set/3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab';
    'Training', 'RW2.mat', '11. Randomized Battery Usage Data Set/3. Battery_Uniform_Distribution_Variable_Charge_Room_Temp/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab';
    
    % Validation
    'Validation', 'B0007.mat', '5. Battery Data Set/1. BatteryAgingARC-FY08Q4';
    
    % Testing
    'Testing', 'B0018.mat', '5. Battery Data Set/1. BatteryAgingARC-FY08Q4';
    'Testing', 'RW9.mat', '11. Randomized Battery Usage Data Set/1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab';
    'Testing', 'RW10.mat', '11. Randomized Battery Usage Data Set/1. Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab';
};

% The user requested remaining files into 'Unused'. 
% For simplicity in this script, we'll focus on processing the targeted ones. 
% We will also handle moving/copying remaining ones by searching for them.

% Process the specified files
for k = 1:size(files_to_process, 1)
    category = files_to_process{k, 1};
    fname = files_to_process{k, 2};
    rel_path = files_to_process{k, 3};
    
    file_path = fullfile(datasets_dir, rel_path, fname);
    out_path = fullfile(data_out_dir, category);
    
    fprintf('Processing %s into %s...\n', fname, category);
    if exist(file_path, 'file')
        process_mat_file(file_path, fname, out_path);
    else
        fprintf('WARNING: File %s not found at %s.\n', fname, file_path);
    end
end

% Search for all other .mat files to put into Unused
disp('Scanning for remaining .mat files to put in Unused...');
all_mats = dir(fullfile(datasets_dir, '**', '*.mat'));
processed_names = files_to_process(:, 2);
for k = 1:length(all_mats)
    fname = all_mats(k).name;
    if ~ismember(fname, processed_names)
        file_path = fullfile(all_mats(k).folder, fname);
        out_path = fullfile(data_out_dir, 'Unused');
        fprintf('Processing Unused file %s...\n', fname);
        process_mat_file(file_path, fname, out_path);
    end
end

disp('Data preprocessing complete!');

%% Helper Functions

function process_mat_file(filepath, filename, outdir)
    % Load the file
    loaded_data = load(filepath);
    [~, name, ~] = fileparts(filename);
    
    % Check if it's Dataset 5 (variable name is the cell name)
    if isfield(loaded_data, name) && isfield(loaded_data.(name), 'cycle')
        cycles = loaded_data.(name).cycle;
        for i = 1:length(cycles)
            % Some cycles don't have enough data or missing fields
            if ~isfield(cycles(i).data, 'Time') || isempty(cycles(i).data.Time)
                continue;
            end
            
            t = cycles(i).data.Time(:);
            
            % Depending on cycle type, fields might be named differently
            if isfield(cycles(i).data, 'Voltage_measured')
                v = cycles(i).data.Voltage_measured(:);
                i_curr = cycles(i).data.Current_measured(:);
                temp = cycles(i).data.Temperature_measured(:);
            elseif isfield(cycles(i).data, 'Voltage_charge')
                v = cycles(i).data.Voltage_charge(:);
                i_curr = cycles(i).data.Current_charge(:);
                temp = cycles(i).data.Temperature_charge(:);
                % If temperature doesn't exist for some reason
                if isempty(temp), temp = zeros(size(v)); end
            else
                continue; % Skip if we can't find voltage/current
            end
            
            % Filter and Export
            export_cycle(name, i, t, v, i_curr, temp, outdir);
        end
        
    % Check if it's Dataset 11 (variable name is 'data' and has 'step')
    elseif isfield(loaded_data, 'data') && isfield(loaded_data.data, 'step')
        steps = loaded_data.data.step;
        for i = 1:length(steps)
            % Check required fields
            if ~isfield(steps(i), 'relativeTime') || isempty(steps(i).relativeTime)
                continue;
            end
            
            t = steps(i).relativeTime(:);
            v = steps(i).voltage(:);
            i_curr = steps(i).current(:);
            temp = steps(i).temperature(:);
            
            % Filter and Export
            export_cycle(name, i, t, v, i_curr, temp, outdir);
        end
    else
        fprintf('Unrecognized structure in %s, skipping.\n', filename);
    end
end

function export_cycle(cell_name, cycle_idx, t, v, i_curr, temp, outdir)
    % Apply zero-phase filtering (filtfilt) to remove noise without shifting
    % Using a moving average window of 5
    windowSize = 5;
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;
    
    % filtfilt requires data length > 3*windowSize
    if length(v) > 3 * windowSize
        v_filt = filtfilt(b, a, v);
        i_filt = filtfilt(b, a, i_curr);
    else
        v_filt = v;
        i_filt = i_curr;
    end
    
    % Flatten data into a matrix
    % Columns: Time, Voltage, Current, Temperature
    % Ensure they are all column vectors of same length
    len = min([length(t), length(v_filt), length(i_filt), length(temp)]);
    data_matrix = [t(1:len), v_filt(1:len), i_filt(1:len), temp(1:len)];
    
    % Write to CSV
    csv_filename = sprintf('%s_cycle%03d.csv', cell_name, cycle_idx);
    csv_path = fullfile(outdir, csv_filename);
    
    % Write header and data
    fid = fopen(csv_path, 'w');
    if fid == -1
        fprintf('Error opening file %s for writing.\n', csv_path);
        return;
    end
    fprintf(fid, 'Time,Voltage,Current,Temperature\n');
    fclose(fid);
    
    dlmwrite(csv_path, data_matrix, '-append', 'precision', 9, 'delimiter', ',');
end
