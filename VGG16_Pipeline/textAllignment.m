function rotationAngle = textAllignment(image)
    angles = -12:0.5:12; % Search in a reasonable range
    [R, ~] = radon(image, angles);
    variance = var(R);
    [~, idx] = max(variance);
    rotationAngle = angles(idx);
    % correctedImg = imrotate(image, -rotationAngle, 'bilinear', 'crop');
end