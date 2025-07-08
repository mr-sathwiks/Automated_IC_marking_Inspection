%% Function to correct rotation of a new image
function correctedImage = correctRotation(originalImage, trainedNet)

    % Resize image to match network input size
    I_resized = imresize(originalImage, [224 224]);
    
    % Ensure image is RGB (3 channels)
    if size(I_resized, 3) == 1
        I_resized = cat(3, I_resized, I_resized, I_resized);
    end
    
    % Predict orientation
    [label, scores] = classify(trainedNet, I_resized);
    
    % Convert label to numeric angle
    predictedAngle = str2double(char(label));
    
    % Check if angle is valid
    if isnan(predictedAngle) || ~isfinite(predictedAngle)
        warning('Invalid angle detected. Using 0 degrees.');
        predictedAngle = 0;
    end
    
    % Correct the orientation by rotating in opposite direction
    correctedImage = imrotate(originalImage, -predictedAngle);
end