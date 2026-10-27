# High-innovation Fault Diagnosis! Multimodal Fusion Classification Model Based on 1D-GRU + 2D-MTF-ResNet-CBAM

Preface
This issue introduces a parallel multimodal fusion classification model based on 1D-GRU+2D-MTF-ResNet-CBAM, which has achieved significant results in fault diagnosis tasks!
1. Model Introduction and Innovation Points
1.1 Model Introduction
The model combines time-frequency images and one-dimensional time series signals, optimized with the CBAM attention mechanism and ResNet for multimodal feature fusion. It is used for fault signal classification, effectively integrating the spatial features of time-frequency images and the temporal sequence features of one-dimensional signals, fully leveraging the advantages of multimodal features.
Dataset: CWRU West China University Bearing Dataset
Environmental Framework: Python 3.9, PyTorch 2.1 or above are required to run
Time-frequency image transformation: Five methods of time-frequency image transformation are provided
Accuracy: 100% accuracy on the test set
Use Cases: For papers requiring network structure explanations or graduation design requirements
Code Guarantee: Code comments are detailed, ready to run without modification.
Note: We also provide accompanying model explanations (for learning network structures) and parameter tuning guidance! Students who need to complete graduation projects or write papers will find this module rich, innovative, and highly performant!!
1.2 Innovation Points Introduction
Innovation One: Multimodal Fusion
This model integrates time-frequency images and one-dimensional time series signals, fully utilizing the complementary nature of these two types of data. Time-frequency images are visualized through Markov transition fields (MTF), capturing the frequency and time characteristics of signals. Meanwhile, one-dimensional time series signals retain the original time-dependent information, suitable for processing using recurrent neural networks (RNN) or GRU. By fusing these two features:
Time-frequency images capture the high-frequency and low-frequency trend changes in signals, aiding in identifying fault features in the frequency domain.
One-dimensional time series signals preserve the time-dependent characteristics of signals, reflecting the dynamic evolution of faults over time.
This dual-channel data fusion enables the model to simultaneously utilize time, frequency, and image features, thereby significantly enhancing the accuracy of fault classification.
Innovation Two: Optimized ResNet with CBAM Attention Mechanism
CBAM (Convolutional Block Attention Module) is an attention mechanism for convolutional neural networks (CNNs), aimed at improving the representation capabilities of the model. CBAM enhances feature representation by combining channel attention and spatial attention to improve model performance.
By inserting CBAM modules after each residual block in ResNet, we balance computational load with performance improvement:
(1) Enhanced feature representation: Combination of channel and spatial attention: CBAM provides two modules: channel attention and spatial attention. These modules enhance feature representation from both the channel dimension and spatial dimension of the feature map. By inserting CBAM into the basic units of ResNet, the network can better capture the importance of different features, enabling effective information extraction and aggregation at deeper layers.
(2) Fine-grained feature selection: Fine-grained feature focusing: CBAM assigns weights to each channel and spatial position of the feature map, allowing for fine-grained selection and emphasis on important features. This mechanism allows ResNet to intelligently ignore irrelevant background information and focus on key features for fault classification.
(3) Seamless integration with residual connections Due to the simplicity and efficiency of CBAM, it can be directly inserted into each residual block of ResNet without significantly increasing computational complexity. This integration method enables ResNet to maintain its original advantages (such as gradient flowability and training stability) while enhancing its feature representation capability.
(4) Enhance Network Performance
(5) Efficiency and Flexibility of Computation
Flexibility: CBAM can be flexibly applied to different variants of ResNet (such as ResNet-18, ResNet-34, ResNet-50, etc.), and can be adjusted and optimized according to the specific needs of the task.
Combining CBAM with ResNet is an effective method to enhance the expressive power of ResNet. This combination not only maintains the depth and width of the network but also strengthens the feature representation capability by introducing attention mechanisms.
Innovation 3: Time-series feature extraction based on one-dimensional signal sequence stacking
When processing one-dimensional time series signals, we stacked the one-dimensional signal sequence data and employed GRU (Gated Recurrent Unit) to extract temporal features. This innovative design has enhanced the computational efficiency of GRU, allowing for more effective extraction of critical features during fault occurrences, significantly improving the accuracy of signal classification.
Innovation 4: The Advantage of Feature Fusion
In the multimodal fusion part of the model, after extracting time-frequency image features and one-dimensional signal features through ResNet-CBAM, we adopt a feature concatenated fusion method to combine the two types of features. Compared with traditional models that only use single mode features, the fused features perform better in classification tasks, mainly reflected in:
The time-frequency image and the time-series signal provide different perspectives of feature information. The former provides frequency domain features, while the latter retains time-dependent characteristics. The combination of the two can more comprehensively reflect the fault characteristics of signals.
Through feature fusion, the model is more robust in capturing fault features under different modes, especially in complex fault signal environments. The fused features can better cope with noise interference and signal changes.
This feature fusion strategy enables our model to maintain high-efficiency and accurate classification performance in various fault modes, enhancing the robustness and generalization capability of the model in practical applications.
2. Data Preprocessing for Bearing Faults
2.1 Import Data
Refer to the previous article for preprocessing of fault classification 10, and the Case Western Reserve University bearing data set: Classification 10 dataset.
\ttrain_set, val_set, test_set are all divided into training set, validation set and test set according to the ratio of 7:2:1, and finally save the data.
The above figure is the form of data reading and preprocessing ideas.
2.2 Data preprocessing, video image transformation
We provide five time-frequency image transformation methods: Markov Transition Function (MTF), Recursive Graph (RP), Gram Matrix (GAF), Continuous Wavelet Transform (CWT), and Short Time Fourier Transform (STFT). These methods can be flexibly replaced with different time-frequency image types from multimodal features.
This article adopts the Markov Transition Field (MTF) as the method for time-frequency image transformation, and the generated time-frequency images are shown as follows:
\t\t  
3. Effect Display of the Innovative Model
(1) Model Training Visualization
(2) Model Evaluation
\t50 epochs, accuracy of 98%, the classification effect is significant with 1D-GRU + 2D-MTF-ResNet-CBAM network.\t The model can fully extract spatial and temporal features of bearing fault signals, converges quickly, has excellent performance, high accuracy, and obvious effect!
(3) Confusion Matrix
\tOther visualizations:
(1) Original data t-SNE feature visualization
(2) t-SNE feature visualization after model training:
Click to download: Complete original data, Python code
Previously Featured Content:
Python Bearing Fault Diagnosis | From Beginner to Publication
Python Bearing Fault Diagnosis (14) High-Innovation Fault Identification Model
| SCI 1 Zone High-Innovation Bearing Fault Diagnosis Model!
High-Innovation Bearing Fault Diagnosis Model Based on GADF+Swin-CNN-GAM
Attention Magic Mod | Superb Bearing Fault Diagnosis Model!
Python Bearing Fault Diagnosis (19) - Innovation in Diagnosis Model Based on Transformer-BiLSTM
Bearing Fault Diagnosis Model Based on K-NN + GCN
27 Bearing Fault Diagnosis Models | The Most Affordable Full Set Online
| Bearing Fault Diagnosis Model Based on KAN and Convolutional Kernels
Time-frequency chart update | Bearing fault diagnosis based on visual top conference + time-frequency chart
Fault Diagnosis | Code Explanation + Live Classes, currently in progress!
Fault Diagnosis | Innovative Model Update: Diagnostic Model Based on SSA-CNN-Transformer
Rapid Document | Parallel Fault Diagnosis Model Based on 2D-SWinTransformer+1D-CNN-SENet
| Kan, Kan convolution combined with attention mechanism!
| Parallel Fault Diagnosis Model Based on 2D-SwinTransformer + BiGRU-GlobalAttention
Positional Encoding Demystified | A Comprehensive Guide to Positional Encoding in Transformers
Resource Sharing | Intelligent Fault Diagnosis and Prediction Based on Graph Neural Networks
Innovation Point | Fault Diagnosis Model Based on Fast Fourier Transform Convolution (FFC)
Code Open Source! | Fault Diagnosis with Domain Adversarial Graph Convolution Network under Varying Conditions
Over Thirty Open-Source Datasets | No More Worries About Datasets for Fault Diagnosis!
Fault Diagnosis | A Must-Have for Graduate Students and Junior Researchers at the Undergraduate Level!
Superior! Top-Level Conference Innovation Integration! Parallel Classification Network Based on 2D-SWinTransformer
END
\t2. Friends who like this can click to follow, give the article a "like", share it on WeChat Moments or discuss in groups.

## Images

![High-innovation fault diagnosis! Multimodal fusion classification model based on 1D-GRU + 2D-MTF-ResNet-CBAM](06.jpg)
![High-innovation fault diagnosis! Multimodal fusion classification model based on 1D-GRU + 2D-MTF-ResNet-CBAM](07.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
