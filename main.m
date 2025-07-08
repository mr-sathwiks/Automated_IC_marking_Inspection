% Script to generate sample result images showing preprocessing steps
% and create separate visualizations for train and test datasets

% Clear workspace
clear all; close all; clc;

% Set paths to your data folders - update these to your current paths
train_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage 3 ResNet\dataset_augmented\train';
test_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage 3 ResNet\dataset_augmented\test';
results_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage 3 ResNet\Results\Report';
sample_results_path = fullfile(results_path, 'sample_results');

% Create separate directories for train and test results
train_results_path = fullfile(sample_results_path, 'train');
test_results_path = fullfile(sample_results_path, 'test');

% Create directories if they don't exist
if ~exist(sample_results_path, 'dir')
    mkdir(sample_results_path);
end
if ~exist(train_results_path, 'dir')
    mkdir(train_results_path);
end
if ~exist(test_results_path, 'dir')
    mkdir(test_results_path);
end

%% Initialize GPU if available
if gpuDeviceCount > 0
    gpu = gpuDevice(1);
    fprintf('Using GPU: %s with %g GB memory\n', gpu.Name, gpu.AvailableMemory/1e9);
else
    warning('No GPU detected. Using CPU.');
end

%% Load models
fprintf('Loading models...\n');
rotation_correction_model = load('rotationCorrectionNet.mat');

%% Load datasets
fprintf('Loading datasets...\n');
train_dataset = init_dataset(train_path);
test_dataset = init_dataset(test_path);

%% Generate sample result images for train dataset
fprintf('\nProcessing TRAINING images...\n');
train_sample_images = generate_sample_images(train_dataset, train_results_path, rotation_correction_model, 'Training');

%% Generate sample result images for test dataset
fprintf('\nProcessing TEST images...\n');
test_sample_images = generate_sample_images(test_dataset, test_results_path, rotation_correction_model, 'Test');

%% Create a combined montage with both train and test samples
fprintf('\nCreating combined train/test montage...\n');

if ~isempty(train_sample_images) && ~isempty(test_sample_images)
    % Select a subset of images from each set for the combined montage
    max_per_set = min(12, min(length(train_sample_images), length(test_sample_images)));
    
    train_subset = train_sample_images(1:min(max_per_set, length(train_sample_images)));
    test_subset = test_sample_images(1:min(max_per_set, length(test_sample_images)));
    
    combined_images = [train_subset, test_subset];
    
    % Create combined montage
    fig = figure('Visible', 'off', 'Position', [100, 100, 2400, 1200]);
    montage(combined_images, 'Size', [4, 6]);
    title('IC Chip Marking Inspection - Train and Test Samples', 'FontSize', 16);
    
    % Add labels for train and test regions
    annotation('textbox', [0.1, 0.95, 0.3, 0.05], ...
        'String', 'TRAINING SAMPLES', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    annotation('textbox', [0.6, 0.95, 0.3, 0.05], ...
        'String', 'TEST SAMPLES', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    % Save combined montage
    combined_path = fullfile(sample_results_path, 'train_test_samples_combined.png');
    saveas(fig, combined_path);
    close(fig);
    
    fprintf('Combined train/test montage created and saved to: %s\n', combined_path);
else
    fprintf('Unable to create combined montage due to missing sample images.\n');
end

fprintf('Sample result images generated and saved to:\n');
fprintf('  Train samples: %s\n', train_results_path);
fprintf('  Test samples: %s\n', test_results_path);

%% Helper function to generate sample images
function sample_images = generate_sample_images(dataset, results_path, rotation_correction_model, set_name)
    % Get unique chip variants and classes
    chip_variants = unique({dataset.chip_variant});
    classes = unique({dataset.class});
    
    % Initialize array to store images for montage
    sample_images = {};
    
    % Iterate through each chip variant and class
    for i = 1:length(chip_variants)
        chip = chip_variants{i};
        
        for j = 1:length(classes)
            class_name = classes{j};
            
            fprintf('Processing %s: %s/%s...\n', set_name, chip, class_name);
            
            % Find all entries for this chip/class combination
            entries = dataset(strcmp({dataset.chip_variant}, chip) & ...
                           strcmp({dataset.class}, class_name));
            
            % Skip if no entries found
            if isempty(entries)
                fprintf('No entries found for %s/%s\n', chip, class_name);
                continue;
            end
            
            % Collect all filenames from all matching entries
            all_filenames = [];
            all_paths = {};
            for k = 1:length(entries)
                for l = 1:length(entries(k).filenames)
                    all_filenames = [all_filenames; entries(k).filenames(l)];
                    all_paths = [all_paths; {entries(k).path}];
                end
            end
            
            % Select 5 random images (or less if not enough available)
            num_samples = min(5, length(all_filenames));
            if length(all_filenames) > 0
                rand_indices = randperm(length(all_filenames), num_samples);
            else
                rand_indices = [];
            end
            
            % Process each selected image
            for m = 1:length(rand_indices)
                idx = rand_indices(m);
                img_path = fullfile(all_paths{idx}, all_filenames(idx).filename);
                [~, filename, ~] = fileparts(all_filenames(idx).filename);
                
                fprintf('  Processing image %d/%d: %s\n', m, num_samples, img_path);
                
                % Load image
                img = imread(img_path);
                
                % Create a figure to show processing steps
                fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
                
                % Original image
                subplot(2, 3, 1);
                imshow(img);
                title('Original Image', 'FontSize', 12);
                
                % Apply rotation correction
                correctedImg = correctRotation(img, rotation_correction_model.trainedNet);
                subplot(2, 3, 2);
                imshow(correctedImg);
                title('Rotation Corrected', 'FontSize', 12);
                
                % Convert to grayscale and apply text alignment
                grayImg = rgb2gray(correctedImg);
                rotationAngle = textAllignment(grayImg);
                textAligned_gray = imrotate(grayImg, -rotationAngle, 'bilinear', 'crop');
                subplot(2, 3, 3);
                imshow(textAligned_gray);
                title(['Text Aligned (' num2str(rotationAngle, '%.2f') '°)'], 'FontSize', 12);
                
                % Binarization
                T = adaptthresh(textAligned_gray, 0.35, 'Statistic', 'gaussian');
                binaryImg = imbinarize(textAligned_gray, T);
                binaryImg_forOCR = imcomplement(binaryImg);
                subplot(2, 3, 4);
                imshow(binaryImg);
                title('Binarized Image', 'FontSize', 12);
                
                % Text detection and recognition
                try
                    % Text detection using CRAFT
                    bboxes = detectTextCRAFT(textAligned_gray, 'CharacterThreshold', 0.3);
                    
                    % OCR on detected regions
                    recognizedText = cell(size(bboxes, 1), 1);
                    for x = 1:size(bboxes, 1)
                        ocrResults = ocr(binaryImg_forOCR, bboxes(x,:), ...
                            'LayoutAnalysis', 'line', ...
                            'CharacterSet', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', ...
                            'Model', 'english');
                        
                        if ~isempty(ocrResults.Text)
                            recognizedText{x} = strtrim(ocrResults.Text);
                        else
                            recognizedText{x} = '';
                        end
                    end
                    
                    % Show text detection and recognition
                    subplot(2, 3, 5);
                    imshow(textAligned_gray);
                    hold on;
                    for x = 1:size(bboxes, 1)
                        rectangle('Position', bboxes(x,:), 'EdgeColor', 'r', 'LineWidth', 2);
                        if ~isempty(recognizedText{x})
                            text(bboxes(x,1), bboxes(x,2)-5, recognizedText{x}, ...
                                'Color', 'green', 'FontWeight', 'bold', 'FontSize', 12);
                        end
                    end
                    title('Text Detection & OCR', 'FontSize', 12);
                    hold off;
                    
                    % Combine recognized text
                    text_content = strjoin(recognizedText(~cellfun(@isempty, recognizedText)), ' ');
                    
                catch e
                    % If text detection fails
                    fprintf('  Text detection failed: %s\n', e.message);
                    subplot(2, 3, 5);
                    imshow(textAligned_gray);
                    title('Text Detection Failed', 'FontSize', 12);
                    text_content = '';
                end
                
                % Final results display
                subplot(2, 3, 6);
                imshow(correctedImg);
                
                % Add class information
                title(['Class: ' class_name], 'FontSize', 12);
                
                % Add recognized text directly to the image
                if ~isempty(text_content)
                    t = text(10, size(correctedImg, 1) - 20, ['Text: ' text_content], ...
                        'Color', 'white', 'BackgroundColor', [0 0 0 0.5], ...
                        'FontSize', 10, 'FontWeight', 'bold');
                else
                    t = text(10, size(correctedImg, 1) - 20, 'No text detected', ...
                        'Color', 'white', 'BackgroundColor', [0 0 0 0.5], ...
                        'FontSize', 10, 'FontWeight', 'bold');
                end
                
                % Add a main title with chip and file information
                sgtitle(sprintf('%s: %s - %s - %s', set_name, chip, class_name, filename), 'FontSize', 14);
                
                % Render figure to image
                set(fig, 'PaperPositionMode', 'auto');
                frame = getframe(fig);
                sample_img = frame.cdata;
                
                % Save individual image
                sample_filename = sprintf('%s_%s_%s.png', chip, class_name, filename);
                sample_path = fullfile(results_path, sample_filename);
                imwrite(sample_img, sample_path);
                
                % Store image for montage
                sample_images{end+1} = sample_img;
                
                % Close figure to free memory
                close(fig);
                
                % Reset GPU if needed
                if gpuDeviceCount > 0 && mod(m, 5) == 0
                    reset(gpuDevice(1));
                end
            end
        end
    end
    
    %% Create montages for this dataset
    fprintf('Creating montages for %s set with %d sample images...\n', set_name, length(sample_images));
    
    if ~isempty(sample_images)
        % Create multiple montages with max 25 images each for better visibility
        num_montages = ceil(length(sample_images) / 25);
        
        for montage_idx = 1:num_montages
            % Calculate start and end indices for this montage
            start_idx = (montage_idx - 1) * 25 + 1;
            end_idx = min(montage_idx * 25, length(sample_images));
            
            % Get images for this montage
            montage_images = sample_images(start_idx:end_idx);
            
            % Create figure for montage
            fig = figure('Visible', 'off', 'Position', [100, 100, 1800, 1200]);
            
            % Create montage
            montage(montage_images, 'Size', [5, 5]);
            title(sprintf('IC Chip Marking Inspection - %s Sample Results (Part %d of %d)', ...
                set_name, montage_idx, num_montages), 'FontSize', 16);
            
            % Save montage
            montage_path = fullfile(results_path, sprintf('sample_montage_%s_part%d.png', set_name, montage_idx));
            saveas(fig, montage_path);
            close(fig);
            
            fprintf('  Montage part %d created and saved.\n', montage_idx);
        end
        
        % Create a single combined montage with all images (might be large)
        fig = figure('Visible', 'off', 'Position', [100, 100, 2400, 1600]);
        montage(sample_images);
        title(sprintf('IC Chip Marking Inspection - All %s Sample Results', set_name), 'FontSize', 16);
        combined_montage_path = fullfile(results_path, sprintf('all_samples_%s_combined.png', set_name));
        saveas(fig, combined_montage_path);
        close(fig);
        
        fprintf('Combined montage created and saved to: %s\n', combined_montage_path);
    else
        fprintf('No sample images generated for %s set, cannot create montage.\n', set_name);
    end
end
