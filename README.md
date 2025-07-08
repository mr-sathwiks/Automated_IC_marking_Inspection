# Automated_IC_marking_Inspection

🔍 Project Overview:
In IC manufacturing, ensuring the quality of printed markings is critical for traceability and quality control. Manual inspection is time-consuming and error-prone, so I developed a fully automated system that classifies chips as “Good” or “Defective” based on their markings.

🛠️ Key Features:
Advanced Preprocessing: Major rotation correction (VGG16-based, 99.1% accuracy), noise removal, adaptive binarisation, and fine text alignment.
Text Region Segmentation: Used the CRAFT model to accurately localise text regions.
OCR & Text Feature Extraction: Applied OCR to extract marking content, encoded as bag-of-words vectors.
Deep Feature Extraction: Leveraged ResNet50 to extract robust visual features from each chip image.
Feature Fusion: Combined image and text features for comprehensive defect detection.
Classification: Trained a GentleBoost ensemble with a cost matrix to prioritise defect detection, achieving up to 85% accuracy on test data.

📊 Results:
High accuracy and generalisation on unseen IC types. (Accuracy: 85%)
Robust to orientation, noise, and marking variability.
Modular and scalable for real-world deployment.

🔬 Tech Stack:
MATLAB, Deep Learning Toolbox, CRAFT, OCR, VGG16, ResNet50, GentleBoost
