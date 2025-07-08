function [result_img, text_content] = visualizeResult(img, rotationNet, predicted_label, true_label)
    % Process image
    correctedImg = correctRotation(img, rotationNet);
    grayImg = rgb2gray(correctedImg);
    rotationAngle = textAllignment(grayImg);
    textAligned_gray = imrotate(grayImg, -rotationAngle, 'bilinear', 'crop');
    
    % Binarization
    T = adaptthresh(textAligned_gray, 0.35, 'Statistic', 'gaussian');
    binaryImg = imbinarize(textAligned_gray, T);
    
    % Text detection
    text_content = '';
    try
        bboxes = detectTextCRAFT(textAligned_gray);
        ocr_results = ocr(binaryImg, bboxes, 'TextLayout', 'block');
        words = ocr_results.Words;
        text_content = strjoin(words(~cellfun(@isempty, words)), ' ');
        
        % Create visualization
        result_img = correctedImg;
        result_img = insertObjectAnnotation(result_img, 'rectangle', ...
                                          bboxes, 'Text', 'LineWidth', 2, 'Color', 'blue');
        
        % Add classification result
        label_str = sprintf('True: %s | Pred: %s', ...
                           char(true_label), char(predicted_label));
        
        % Determine color based on correct/incorrect prediction
        if predicted_label == true_label
            label_color = [0, 0.7, 0]; % Green for correct
        else
            label_color = [0.9, 0, 0]; % Red for incorrect
        end
        
        % Add text at the top of the image
        result_img = insertText(result_img, [10, 10], label_str, ...
                               'BoxColor', label_color, 'BoxOpacity', 0.6, ...
                               'TextColor', 'white', 'FontSize', 14);
    catch
        % If text detection fails, use original image with just the label
        result_img = correctedImg;
        label_str = sprintf('True: %s | Pred: %s (Text Detection Failed)', ...
                           char(true_label), char(predicted_label));
        
        % Determine color based on correct/incorrect prediction
        if predicted_label == true_label
            label_color = [0, 0.7, 0]; % Green for correct
        else
            label_color = [0.9, 0, 0]; % Red for incorrect
        end
        
        % Add text at the top of the image
        result_img = insertText(result_img, [10, 10], label_str, ...
                               'BoxColor', label_color, 'BoxOpacity', 0.6, ...
                               'TextColor', 'white', 'FontSize', 14);
    end
end
