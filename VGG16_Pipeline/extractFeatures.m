function [img_features, text_content] = extractFeatures(img, net, rotationCorrectionNet)
    % Apply preprocessing pipeline
    % 1. Rotation Correction
    correctedImg = correctRotation(img, rotationCorrectionNet);
    
    % 2. Edge Detection and Binarization
    grayImg = rgb2gray(correctedImg);
    
    % 3. Text Alignment
    rotationAngle = textAllignment(grayImg);
    textAligned_gray = imrotate(grayImg, -rotationAngle, 'bilinear', 'crop');
    
    % 4. Binarization
    T = adaptthresh(textAligned_gray, 0.35, 'Statistic', 'gaussian');
    binaryImg = imbinarize(textAligned_gray, T);
    binaryImg = imcomplement(binaryImg);
    
    % Extract image features using VGG16
    img_resized = imresize(correctedImg, [224 224]);
    if size(img_resized, 3) == 1
        img_resized = cat(3, img_resized, img_resized, img_resized);
    end
    
    % Extract features from fc7 layer using GPU if available
    % Note: For VGG16, we use 'fc7' instead of 'avg_pool'
    if gpuDeviceCount > 0
        img_resized_gpu = gpuArray(img_resized);
        img_features = gather(activations(net, img_resized_gpu, 'fc7', 'OutputAs', 'rows'));
    else
        img_features = activations(net, img_resized, 'fc7', 'OutputAs', 'rows');
    end
    
    img_features = double(img_features);
    
    % Text detection using CRAFT
    try
        % MATLAB's built-in CRAFT text detection
        bboxes = detectTextCRAFT(textAligned_gray, CharacterThreshold=0.3);
        
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
        
        if isempty(text_content)
            text_content = '';
        end
    catch
        % If OCR fails, return empty string
        text_content = '';
    end
end
