function split_dataset(inputDir, outputDir, trainRatio)
    % This function splits a dataset into training and testing sets
    % with true randomization for each IC type and subfolder
    
    % Create output directories
    trainDir = fullfile(outputDir, 'train');
    testDir = fullfile(outputDir, 'test');
    
    if ~exist(trainDir, 'dir')
        mkdir(trainDir);
    end
    
    if ~exist(testDir, 'dir')
        mkdir(testDir);
    end
    
    % Get all IC type folders from input directory
    icFolders = dir(inputDir);
    icFolders = icFolders([icFolders.isdir]);
    icFolders = icFolders(~ismember({icFolders.name}, {'.', '..'}));
    
    % Process each IC type folder
    for i = 1:length(icFolders)
        icFolder = icFolders(i).name;
        fprintf('Processing IC type: %s\n', icFolder);
        
        % Create IC folders in train and test directories
        if ~exist(fullfile(trainDir, icFolder), 'dir')
            mkdir(fullfile(trainDir, icFolder));
        end
        
        if ~exist(fullfile(testDir, icFolder), 'dir')
            mkdir(fullfile(testDir, icFolder));
        end
        
        % Get "Good" and "Defective" subfolders
        subfolders = dir(fullfile(inputDir, icFolder));
        subfolders = subfolders([subfolders.isdir]);
        subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
        
        % Process each subfolder (Good/Defective)
        for j = 1:length(subfolders)
            subfolder = subfolders(j).name;
            fprintf('  Processing subfolder: %s\n', subfolder);
            
            % Create subfolders in train and test directories
            if ~exist(fullfile(trainDir, icFolder, subfolder), 'dir')
                mkdir(fullfile(trainDir, icFolder, subfolder));
            end
            
            if ~exist(fullfile(testDir, icFolder, subfolder), 'dir')
                mkdir(fullfile(testDir, icFolder, subfolder));
            end
            
            % Get all images in the current subfolder
            imageFiles = dir(fullfile(inputDir, icFolder, subfolder, '*.jpg'));
            numFiles = length(imageFiles);
            
            % Sort files by name for consistent initial ordering
            [~, sortOrder] = sort({imageFiles.name});
            imageFiles = imageFiles(sortOrder);
            
            % Use true randomization with a different seed each time
            rng('shuffle');  % This ensures different random numbers each run
            
            % Generate random indices for the entire set of images
            randIndices = randperm(numFiles);
            
            % Determine split point (70% for training)
            numTrainFiles = round(numFiles * trainRatio);
            
            % Split indices into training and testing sets
            trainIndices = randIndices(1:numTrainFiles);
            testIndices = randIndices(numTrainFiles+1:end);
            
            % Copy files to training folder using random indices
            for k = 1:length(trainIndices)
                idx = trainIndices(k);
                sourceFile = fullfile(imageFiles(idx).folder, imageFiles(idx).name);
                destFile = fullfile(trainDir, icFolder, subfolder, imageFiles(idx).name);
                copyfile(sourceFile, destFile);
            end
            
            % Copy files to testing folder using random indices
            for k = 1:length(testIndices)
                idx = testIndices(k);
                sourceFile = fullfile(imageFiles(idx).folder, imageFiles(idx).name);
                destFile = fullfile(testDir, icFolder, subfolder, imageFiles(idx).name);
                copyfile(sourceFile, destFile);
            end
            
            fprintf('    Split: %d training, %d testing images\n', numTrainFiles, numFiles-numTrainFiles);
        end
    end
    
    fprintf('Dataset split completed.\n');
    fprintf('Training set: %d%%\n', round(trainRatio*100));
    fprintf('Testing set: %d%%\n', round((1-trainRatio)*100));
end
