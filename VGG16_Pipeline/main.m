% IC Chip Marking Inspection Pipeline - VGG16 Implementation
% Clear workspace
% clear all; close all; clc;

% Set paths to your data folders
train_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage-3 VGG16\dataset_augmented\train';
test_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage-3 VGG16\dataset_augmented\test';
results_path = 'C:\Users\Sathwik_PC\Box\MSEE\Sem 1\DIP\DIP Project\Stage-3 VGG16\Results';

% Create results directory if it doesn't exist
if ~exist(results_path, 'dir')
    mkdir(results_path);
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
feature_extractor = vgg16;

%% Load dataset using your custom init_dataset function
fprintf('Loading datasets...\n');
train_dataset = init_dataset(train_path);
test_dataset = init_dataset(test_path);

%% Feature extraction and classification
% First, process all training text to build a shared vocabulary
all_train_texts = {};
fprintf('First pass: collecting all training text for vocabulary building...\n');

% Process each entry in training dataset to collect text
for i = 1:length(train_dataset)
    entry = train_dataset(i);
    fprintf('Collecting text from %s/%s (%d images)...\n', entry.chip_variant, entry.class, length(entry.filenames));
    
    for j = 1:length(entry.filenames)
        % Load image
        img_path = fullfile(entry.path, entry.filenames(j).filename);
        img = imread(img_path);
        
        % Apply preprocessing for text extraction
        correctedImg = correctRotation(img, rotation_correction_model.trainedNet);
        grayImg = rgb2gray(correctedImg);
        rotationAngle = textAllignment(grayImg);
        textAligned_gray = imrotate(grayImg, -rotationAngle, 'bilinear', 'crop');
        
        % Text detection
        try
            bboxes = detectTextCRAFT(textAligned_gray, CharacterThreshold=0.3);
            T = adaptthresh(textAligned_gray, 0.35, 'Statistic', 'gaussian');
            binaryImg = imbinarize(textAligned_gray, T);
            binaryImg = imcomplement(binaryImg);
            
            recognizedText = cell(size(bboxes, 1), 1);
            for x = 1:size(bboxes,1)
                ocrResults = ocr(binaryImg, bboxes(x,:), ...
                                LayoutAnalysis="line", ...
                                CharacterSet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', ...
                                Model= 'english');
                if ~isempty(ocrResults.Text)
                    recognizedText{x} = strtrim(ocrResults.Text);
                else
                    recognizedText{x} = '';
                end
            end
            
            words = recognizedText;
            text_content = strjoin(words(~cellfun(@isempty, words)), ' ');
            
            if ~isempty(text_content)
                all_train_texts{end+1} = text_content;
            end
        catch
            % If text detection fails, continue
            continue;
        end
        
        % Reset GPU periodically
        if gpuDeviceCount > 0 && mod(j, 10) == 0
            reset(gpuDevice(1));
        end
    end
end

% Build shared vocabulary from all training texts
fprintf('Building shared vocabulary from %d text samples...\n', length(all_train_texts));
docs_train = tokenizedDocument(all_train_texts);
shared_bag = bagOfWords(docs_train);

% Second pass: extract features using shared vocabulary
fprintf('Second pass: extracting features using shared vocabulary...\n');

% 1. Train set processing
train_features = [];
train_labels = categorical([]);
train_chip_types = {};

% Process each entry in training dataset
for i = 1:length(train_dataset)
    entry = train_dataset(i);
    fprintf('Processing %s/%s (%d images)...\n', entry.chip_variant, entry.class, length(entry.filenames));
    
    for j = 1:length(entry.filenames)
        % Load image
        img_path = fullfile(entry.path, entry.filenames(j).filename);
        img = imread(img_path);
        
        % Extract features
        [img_features, text_content] = extractFeatures(img, feature_extractor, rotation_correction_model.trainedNet);
        
        % Encode text with shared vocabulary
        txt_features = encodeTextFeatures(text_content, shared_bag);
        
        % Combine features
        combined_features = [img_features, txt_features];
        
        % Store features and label
        train_features = [train_features; combined_features];
        train_labels = [train_labels; categorical({entry.class})];
        train_chip_types = [train_chip_types; entry.chip_variant];
        
        % Reset GPU periodically
        if gpuDeviceCount > 0 && mod(j, 10) == 0
            reset(gpuDevice(1));
        end
    end
end

% 2. Test set processing
test_features = [];
test_labels = categorical([]);
test_chip_types = {};

% Process each entry in test dataset
for i = 1:length(test_dataset)
    entry = test_dataset(i);
    fprintf('Processing %s/%s (%d images)...\n', entry.chip_variant, entry.class, length(entry.filenames));
    
    for j = 1:length(entry.filenames)
        % Load image
        img_path = fullfile(entry.path, entry.filenames(j).filename);
        img = imread(img_path);
        
        % Extract features
        [img_features, text_content] = extractFeatures(img, feature_extractor, rotation_correction_model.trainedNet);
        
        % Encode text with shared vocabulary - USING THE SAME BAG AS TRAINING
        txt_features = encodeTextFeatures(text_content, shared_bag);
        
        % Combine features
        combined_features = [img_features, txt_features];
        
        % Store features and label
        test_features = [test_features; combined_features];
        test_labels = [test_labels; categorical({entry.class})];
        test_chip_types = [test_chip_types; entry.chip_variant];
        
        % Reset GPU periodically
        if gpuDeviceCount > 0 && mod(j, 10) == 0
            reset(gpuDevice(1));
        end
    end
end

% Verify dimensions match
fprintf('Train features size: %d x %d\n', size(train_features, 1), size(train_features, 2));
if ~isempty(test_features)
    fprintf('Test features size: %d x %d\n', size(test_features, 1), size(test_features, 2));
    if size(train_features, 2) ~= size(test_features, 2)
        error('Feature dimension mismatch between train (%d) and test (%d)', ...
              size(train_features, 2), size(test_features, 2));
    end
end

%% Configure model training
fprintf('Configuring model training...\n');
t = templateTree('MaxNumSplits', 50, 'MinLeafSize', 5);
max_trees = 1000; % Reduced for faster training
fold_count = 10; % 10 fold cross-validation

%% Train and evaluate model
fprintf('Training model with %d fold(s)...\n', fold_count);

if fold_count == 1
    % Fast single train-validation split
    fprintf('Using single train-validation split for rapid development...\n');
    
    % Simple random split
    cv = cvpartition(train_labels, 'HoldOut', 0.2);
    train_idx = cv.training;
    val_idx = cv.test;
    
    % Train model with printouts every 20 trees
    mdl = fitcensemble(train_features(train_idx,:), train_labels(train_idx), ...
                      'Method', 'GentleBoost', ...
                      'Learners', t, ...
                      'NumLearningCycles', max_trees, ...
                      'LearnRate', 0.1, ...
                      'Cost', [0 1; 5 0], ...
                      'NPrint', 20); % Will show progress every 20 trees
    
    % Predict on validation set
    val_predictions = predict(mdl, train_features(val_idx,:));
    
    % Compute metrics
    val_accuracy = mean(val_predictions == train_labels(val_idx)) * 100;
    val_cm = confusionmat(train_labels(val_idx), val_predictions, 'Order', categories(train_labels));
    
    % Calculate precision, recall, F1
    val_precision = diag(val_cm) ./ sum(val_cm, 2);
    val_recall = diag(val_cm) ./ sum(val_cm, 1)';
    val_f1 = 2 * (val_precision .* val_recall) ./ (val_precision + val_recall);
    
    fprintf('\nValidation Results:\n');
    disp(array2table(val_cm, ...
                    'VariableNames', strcat('Pred_', categories(train_labels)), ...
                    'RowNames', strcat('True_', categories(train_labels))));
    fprintf('Accuracy: %.2f%%\n', val_accuracy);
    fprintf('F1-score: %.4f\n', mean(val_f1, 'omitnan'));
    
    % Use this model as the final model
    final_model = mdl;
    opt_trees = max_trees;
else
    % Regular cross-validation
    fprintf('Performing %d-fold cross-validation...\n', fold_count);
    
    % No parallel processing option for LogitBoost
    cv_mdl = fitcensemble(train_features, train_labels, ...
                         'Method', 'GentleBoost', ...
                         'Learners', t, ...
                         'NumLearningCycles', max_trees, ...
                         'LearnRate', 0.1, ...
                         'Cost', [0 1; 5 0], ...
                         'CrossVal', 'on', ...
                         'KFold', fold_count, ...
                         'NPrint', 20);
    
    % Get cross-validation predictions
    train_predictions_cv = kfoldPredict(cv_mdl);
    
    % Compute metrics by chip type
    fprintf('\nPer-chip %d-fold CV results:\n', fold_count);
    chip_types = unique(train_chip_types);
    
    for i = 1:length(chip_types)
        chip = chip_types{i};
        fprintf('\n=== %s ===\n', chip);
        
        % Get indices for this chip type
        idx = strcmp(train_chip_types, chip);
        
        % Get actual and predicted labels
        actual = train_labels(idx);
        predicted = train_predictions_cv(idx);
        
        % Compute confusion matrix and metrics
        cm = confusionmat(actual, predicted, 'Order', categories(actual));
        accuracy = sum(diag(cm)) / sum(cm(:)) * 100;
        precision = diag(cm) ./ sum(cm, 2);
        recall = diag(cm) ./ sum(cm, 1)';
        f1_score = 2 * (precision .* recall) ./ (precision + recall);
        
        % Display metrics
        disp(array2table(cm, ...
                        'VariableNames', strcat('Pred_', categories(actual)), ...
                        'RowNames', strcat('True_', categories(actual))));
        fprintf('Accuracy: %.2f%%\n', accuracy);
        fprintf('F1-score: %.4f\n', mean(f1_score, 'omitnan'));
    end
    
    % Find optimal number of trees
    fprintf('Finding optimal number of trees...\n');
    loss_curve = kfoldLoss(cv_mdl, 'Mode', 'Cumulative');
    [~, opt_trees] = min(loss_curve);
    fprintf('Optimal number of trees: %d\n', opt_trees);
    
    % Train final model with optimal number of trees
    fprintf('Training final model with %d trees...\n', opt_trees);
    final_model = fitcensemble(train_features, train_labels, ...
                              'Method', 'GentleBoost', ...
                              'Learners', t, ...
                              'NumLearningCycles', opt_trees, ...
                              'LearnRate', 0.1, ...
                              'Cost', [0 1; 5 0], ...
                              'NPrint', 20);
end

% Save the model and bag
save(fullfile(results_path, 'final_model_vgg16.mat'), 'final_model', 'shared_bag');

%% Evaluate on test set
if ~isempty(test_features)
    fprintf('Evaluating on test set...\n');
    test_predictions = predict(final_model, test_features);
    
    % Calculate overall metrics
    overall_accuracy = mean(test_predictions == test_labels) * 100;
    overall_cm = confusionmat(test_labels, test_predictions, 'Order', categories(test_labels));
    
    % Calculate precision, recall, F1
    overall_precision = diag(overall_cm) ./ sum(overall_cm, 2);
    overall_recall = diag(overall_cm) ./ sum(overall_cm, 1)';
    overall_f1 = 2 * (overall_precision .* overall_recall) ./ (overall_precision + overall_recall);
    
    % Display overall results
    fprintf('\nOverall Test Results:\n');
    disp(array2table(overall_cm, ...
                    'VariableNames', strcat('Pred_', categories(test_labels)), ...
                    'RowNames', strcat('True_', categories(test_labels))));
    fprintf('Accuracy: %.2f%%\n', overall_accuracy);
    fprintf('F1-score: %.4f\n', mean(overall_f1, 'omitnan'));
end

%% Export visual results of the classification
if ~isempty(test_features)
    fprintf('Generating and saving result visualizations...\n');
    results_img_dir = fullfile(results_path, 'result_images');
    
    % Create directory for result images if it doesn't exist
    if ~exist(results_img_dir, 'dir')
        mkdir(results_img_dir);
    end
    
    % Process a subset of test images (e.g., 10 per chip type) for visualization
    chip_types = unique(test_chip_types);
    
    for i = 1:length(chip_types)
        chip = chip_types{i};
        
        % Get indices for this chip type
        chip_idx = find(strcmp(test_chip_types, chip));
        
        % Select up to 10 samples per chip type
        num_samples = min(10, length(chip_idx));
        sample_idx = chip_idx(randperm(length(chip_idx), num_samples));
        
        for j = 1:length(sample_idx)
            idx = sample_idx(j);
            
            % Find the corresponding entry in test_dataset to get the file
            for k = 1:length(test_dataset)
                if strcmp(test_dataset(k).chip_variant, chip) && ...
                   strcmp(test_dataset(k).class, char(test_labels(idx)))
                    
                    % Get a random image from this entry
                    file_idx = randi(length(test_dataset(k).filenames));
                    img_path = fullfile(test_dataset(k).path, test_dataset(k).filenames(file_idx).filename);
                    img = imread(img_path);
                    
                    % Create visualization
                    [img_processed, text_content] = visualizeResult(img, ...
                                                                   rotation_correction_model.trainedNet, ...
                                                                   test_predictions(idx), test_labels(idx));
                    
                    % Save visualization
                    [~, filename, ~] = fileparts(test_dataset(k).filenames(file_idx).filename);
                    result_filename = sprintf('%s_%s_pred_%s.png', ...
                                             chip, filename, char(test_predictions(idx)));
                    imwrite(img_processed, fullfile(results_img_dir, result_filename));
                    break;
                end
            end
        end
    end
end

%% Save confusion matrices
% For validation confusion matrix (single-fold case)
if fold_count == 1
    figure('Visible', 'off');
    cm_vis = confusionchart(val_cm, categories(train_labels));
    cm_vis.Title = 'Validation Confusion Matrix';
    cm_vis.RowSummary = 'row-normalized';
    cm_vis.ColumnSummary = 'column-normalized';
    saveas(gcf, fullfile(results_path, 'validation_confusion_matrix.png'));
    close(gcf);
end

% For per-chip confusion matrices (cross-validation case)
if fold_count > 1
    for i = 1:length(chip_types)
        chip = chip_types{i};
        
        % Get indices for this chip type
        idx = strcmp(train_chip_types, chip);
        
        % Get actual and predicted labels
        actual = train_labels(idx);
        predicted = train_predictions_cv(idx);
        
        % Create and save confusion matrix
        figure('Visible', 'off');
        cm_vis = confusionchart(actual, predicted);
        cm_vis.Title = sprintf('Confusion Matrix - %s', chip);
        cm_vis.RowSummary = 'row-normalized';
        cm_vis.ColumnSummary = 'column-normalized';
        saveas(gcf, fullfile(results_path, sprintf('cv_confusion_matrix_%s.png', chip)));
        close(gcf);
    end
end

% For test set confusion matrix
if ~isempty(test_features)
    figure('Visible', 'off');
    cm_vis = confusionchart(test_labels, test_predictions);
    cm_vis.Title = 'Test Set Confusion Matrix';
    cm_vis.RowSummary = 'row-normalized';
    cm_vis.ColumnSummary = 'column-normalized';
    saveas(gcf, fullfile(results_path, 'test_confusion_matrix.png'));
    close(gcf);
    
    % Save per-chip test confusion matrices
    for i = 1:length(chip_types)
        chip = chip_types{i};
        
        % Get indices for this chip type in test set
        idx = strcmp(test_chip_types, chip);
        
        if sum(idx) > 0 % Only if we have samples for this chip type
            % Create and save confusion matrix
            figure('Visible', 'off');
            cm_vis = confusionchart(test_labels(idx), test_predictions(idx));
            cm_vis.Title = sprintf('Test Confusion Matrix - %s', chip);
            cm_vis.RowSummary = 'row-normalized';
            cm_vis.ColumnSummary = 'column-normalized';
            saveas(gcf, fullfile(results_path, sprintf('test_confusion_matrix_%s.png', chip)));
            close(gcf);
        end
    end
end

%% Export performance metrics to CSV
% Create a table for overall performance
if ~isempty(test_features)
    overall_metrics = table();
    overall_metrics.ChipVariant = {'Overall'};
    overall_metrics.Accuracy = overall_accuracy;
    overall_metrics.Precision = mean(overall_precision, 'omitnan');
    overall_metrics.Recall = mean(overall_recall, 'omitnan');
    overall_metrics.F1Score = mean(overall_f1, 'omitnan');
    
    % Add per-chip metrics
    for i = 1:length(chip_types)
        chip = chip_types{i};
        
        % Get indices for this chip type in test set
        idx = strcmp(test_chip_types, chip);
        
        if sum(idx) > 0 % Only if we have samples for this chip type
            % Calculate metrics
            chip_acc = mean(test_predictions(idx) == test_labels(idx)) * 100;
            chip_cm = confusionmat(test_labels(idx), test_predictions(idx), ...
                                  'Order', categories(test_labels));
            
            % Handle case where there's only one class in this chip type
            if size(chip_cm, 1) == 1
                chip_precision = 1; % If only one class and all correct
                chip_recall = 1; % If only one class and all correct
            else
                chip_precision = diag(chip_cm) ./ sum(chip_cm, 2);
                chip_recall = diag(chip_cm) ./ sum(chip_cm, 1)';
            end
            
            chip_f1 = 2 * (chip_precision .* chip_recall) ./ (chip_precision + chip_recall);
            
            % Add to table
            row = table();
            row.ChipVariant = {chip};
            row.Accuracy = chip_acc;
            row.Precision = mean(chip_precision, 'omitnan');
            row.Recall = mean(chip_recall, 'omitnan');
            row.F1Score = mean(chip_f1, 'omitnan');
            
            overall_metrics = [overall_metrics; row];
        end
    end
    
    % Save metrics to CSV
    writetable(overall_metrics, fullfile(results_path, 'performance_metrics_vgg16.csv'));
    fprintf('Performance metrics saved to %s\n', fullfile(results_path, 'performance_metrics_vgg16.csv'));
    
    % Also export the confusion matrix as a CSV
    writematrix(overall_cm, fullfile(results_path, 'confusion_matrix_vgg16.csv'));
end

fprintf('Processing complete. Results saved to %s\n', results_path);
