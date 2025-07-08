% This function initializes the dataset to the dataset structure as per the
% folder structure it has been saved in.
function dataset_struct = init_dataset(data_path)
    % Initialize dataset structure based on folder organization
    fprintf('Processing dataset structure from %s...\n', data_path);
    
    % Initialize dataset structure
    dataset_struct = struct('chip_variant', {}, 'class', {}, 'path', {}, 'filenames', {});
    
    % Get top-level folders (chip variants)
    folders = dir(fullfile(data_path, '*'));
    folders = folders([folders.isdir]);
    folders = folders(~ismember({folders.name}, {'.', '..'}));
    
    % Process each folder
    for i = 1:length(folders)
        folder_name = folders(i).name;
        
        % Handle subfolder structure (e.g., Chip1/Good, Chip1/Defective)
        subfolders = dir(fullfile(data_path, folder_name, '*'));
        subfolders = subfolders([subfolders.isdir]);
        subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
        
        if ~isempty(subfolders)
            % Dataset has chip/class subfolder structure
            chip_variant = folder_name;
            
            for j = 1:length(subfolders)
                class_folder = subfolders(j).name;
                
                % Create entry in dataset structure
                entry = struct();
                entry.chip_variant = chip_variant;
                entry.class = class_folder;  % 'Good' or 'Defective'
                entry.path = fullfile(data_path, chip_variant, class_folder);
                
                % Get all image files in this class folder
                image_files = dir(fullfile(entry.path, '*.jpg'));
                image_files = [image_files; dir(fullfile(entry.path, '*.png'))];
                
                file_structs = struct('filename', {}, 'majorRotationAngle', {}, 'minorRotationAngle', {});
                
                for k = 1:length(image_files)
                    file_struct = struct();
                    file_struct.filename = image_files(k).name;
                    file_struct.majorRotationAngle = 0;  % Initialize to 0
                    file_struct.minorRotationAngle = 0;  % Initialize to 0
                    
                    % Add to file_structs array
                    file_structs(k) = file_struct;
                end

                entry.filenames = file_structs;
                
                % Add to dataset structure
                dataset_struct = [dataset_struct; entry];
                
                % fprintf('  Added %s/%s: %d images\n', chip_variant, class_folder, length(entry.filenames));
            end
        end
    end
    
    % Summary
    unique_chips = unique({dataset_struct.chip_variant});
    unique_classes = unique({dataset_struct.class});
    total_images = sum(cellfun(@length, {dataset_struct.filenames}));
    
    fprintf('Dataset summary from %s:\n', data_path);
    fprintf('  Found %d entries across %d chip variants\n', length(dataset_struct), length(unique_chips));
    fprintf('  Classes: %s\n', strjoin(unique_classes, ', '));
    fprintf('  Total images: %d\n', total_images);
end