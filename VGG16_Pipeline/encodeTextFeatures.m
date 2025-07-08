function encoded_features = encodeTextFeatures(text_content, shared_bag)
    if isempty(text_content)
        encoded_features = zeros(1, 100);
        return;
    end
    
    text_doc = tokenizedDocument(text_content);
    
    % Encode with shared vocabulary
    try
        txt_features = double(full(encode(shared_bag, text_doc)));
        
        % Ensure consistent feature length
        if length(txt_features) > 100
            encoded_features = txt_features(1:100);
        elseif length(txt_features) < 100
            encoded_features = [txt_features, zeros(1, 100 - length(txt_features))];
        else
            encoded_features = txt_features;
        end
    catch
        encoded_features = zeros(1, 100);
    end
end

