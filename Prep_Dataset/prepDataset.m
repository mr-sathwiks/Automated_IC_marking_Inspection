% This script first splits the dataset into training and testing sets,
% then applies augmentation ONLY to the training set while maintaining the folder structure.

% Path configurations
inputDir = "C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\dataset"; % Original dataset
splitOutputDir = "C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage 3 ResNet\dataset_split"; % Split dataset
finalOutputDir = "C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage 3 ResNet\dataset_augmented"; % Final augmented dataset

% Train/test split ratio
trainRatio = 0.7; % 70% for training, 30% for testing

% Augmentation parameters
numAugmentationsPerImage = 5;
useRotation90s = true;
useSmallRotations = true;
useScaling = true;
useTranslation = true;

% Step 1: Split the dataset into train (70%) and test (30%) sets
split_dataset(inputDir, splitOutputDir, trainRatio);

% Step 2: Create directory for augmented dataset
if ~exist(finalOutputDir, 'dir')
    mkdir(finalOutputDir);
end

% Step 3: Create train and test directories in output
trainOutputDir = fullfile(finalOutputDir, 'train');
testOutputDir = fullfile(finalOutputDir, 'test');
if ~exist(trainOutputDir, 'dir')
    mkdir(trainOutputDir);
end
if ~exist(testOutputDir, 'dir')
    mkdir(testOutputDir);
end

% Step 4: Configure image data augmenter (same as in augment_dataset.m)
augmenterOptions = {};
if useSmallRotations
    augmenterOptions = [augmenterOptions, {'RandRotation', [-10 10]}];
end
if useScaling
    augmenterOptions = [augmenterOptions, {'RandXScale', [0.8 1.1], 'RandYScale', [0.8 1.1]}];
end
if useTranslation
    augmenterOptions = [augmenterOptions, {'RandXTranslation', [-10 10], 'RandYTranslation', [-10 10]}];
end
augmenter = imageDataAugmenter(augmenterOptions{:});

% Step 5: Process train directory with augmentation
trainInputDir = fullfile(splitOutputDir, 'train');
fprintf('Augmenting training set...\n');

% Get all IC type folders
icFolders = dir(trainInputDir);
icFolders = icFolders([icFolders.isdir]);
icFolders = icFolders(~ismember({icFolders.name}, {'.', '..'}));

for i = 1:length(icFolders)
    icFolder = icFolders(i).name;
    outputICDir = fullfile(trainOutputDir, icFolder);
    if ~exist(outputICDir, 'dir')
        mkdir(outputICDir);
    end
    
    % Get "Good" and "Defective" subfolders
    subfolders = dir(fullfile(trainInputDir, icFolder));
    subfolders = subfolders([subfolders.isdir]);
    subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
    
    for j = 1:length(subfolders)
        subfolder = subfolders(j).name;
        outputSubfolderDir = fullfile(outputICDir, subfolder);
        if ~exist(outputSubfolderDir, 'dir')
            mkdir(outputSubfolderDir);
        end
        
        % Get all images in the current subfolder
        imageFiles = dir(fullfile(trainInputDir, icFolder, subfolder, '*.jpg'));
        
        for k = 1:length(imageFiles)
            imageName = imageFiles(k).name;
            imagePath = fullfile(trainInputDir, icFolder, subfolder, imageName);
            img = imread(imagePath);
            
            % Copy original image to output
            imwrite(img, fullfile(outputSubfolderDir, imageName));
            
            % Generate augmentations
            for l = 1:numAugmentationsPerImage
                augImg = img;
                augTypes = {};
                
                if ~isempty(augmenterOptions)
                    augImg = augment(augmenter, augImg);
                    augTypes{end+1} = 'std';
                end
                
                if useRotation90s && rand > 0.5
                    angles = [90, 180, 270];
                    selectedAngle = angles(randi(length(angles)));
                    augImg = imrotate(augImg, selectedAngle);
                    augTypes{end+1} = sprintf('rot%d', selectedAngle);
                end
                
                typeString = strjoin(augTypes, '_');
                augImageName = sprintf('%s_aug%d_%s.jpg', imageName(1:end-4), l, typeString);
                imwrite(augImg, fullfile(outputSubfolderDir, augImageName));
            end
        end
        
        fprintf(' Processed %s/%s: %d images with %d augmentations each\n', ...
            icFolder, subfolder, length(imageFiles), numAugmentationsPerImage);
    end
end

fprintf('Training set augmentation completed successfully!\n');

% Step 6: Copy the test set without augmentation
testInputDir = fullfile(splitOutputDir, 'test');
copyfile(testInputDir, testOutputDir);

disp('Complete dataset splitting and augmentation workflow.');
