Medical Image Classification — Pneumonia Detection from Chest X-Rays
Project Overview

This project builds a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories — NORMAL and PNEUMONIA — as a binary image classification task.

Data Preparation:

Image resizing: All input images were standardized to 256 × 256 × 3, so the network receives a consistent input shape regardless of the original X-ray dimensions.
Normalization: Pixel values were rescaled from the raw 0–255 range down to [0, 1], which helps the network train faster and more stably.
Data augmentation: Applied horizontal flips and slight random rotations (0.1 factor) during training. This artificially increases the variety of the training set and helps the model generalize instead of memorizing specific images.
Model Architecture & Training
CNN structure: Six convolutional blocks, each pairing a Conv2D layer with MaxPooling2D, to progressively extract and downsample features from the X-ray images.
Regularization: Dropout layers were added throughout the network — 0.2 after the pooling steps and a stronger 0.5 right before the output layer — to reduce overfitting by preventing the model from relying too heavily on any single set of neurons.
Optimizer & loss: Trained with the Adam optimizer and SparseCategoricalCrossentropy loss, run for 25 epochs.

Results:
Training vs. validation behavior: The training and validation curves tracked each other reasonably closely after Dropout was added, suggesting the model wasn't just memorizing the training set.
Validation set balance: The data pipeline was adjusted to keep the validation split balanced between the two classes, since an uneven split would have made the validation metrics unreliable.
Test performance: On the held-out test batches, the model produced confident, correct predictions across both classes.

Takeaways:
Dropout made a noticeable difference in closing the gap between training and validation performance — without it, a 6-block CNN on a dataset like this tends to overfit fairly quickly.
Because pneumonia datasets are commonly imbalanced (more pneumonia cases than normal, or vice versa depending on the source), it's worth reporting precision, recall, and a confusion matrix per class in addition to accuracy — this catches cases where the model favors the majority class, the same issue we saw with the bank marketing dataset.
A useful next step would be plotting the training/validation accuracy and loss curves over the 25 epochs to visually confirm the point where the model stabilizes, and to check whether more or fewer epochs would have helped.
