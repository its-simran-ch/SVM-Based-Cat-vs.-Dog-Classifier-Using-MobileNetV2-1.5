# Prodigy_ML_task_03
Implement a support vector machine (SVM) to classify images of cats and dogs .


## Task 3: Implement a support vector machine (SVM) to classify images of cats and dogs .

**Dataset:** https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data
<br>

**--Dataset Structure--**
<br>
The dataset is well-structured, containing two folders:
<br>
:) **train** (for model training)
<br>
:) **test**  (for model evaluation)
<br>
Each folder has two subdirectories— cats and dogs, making it a great dataset for supervised learning tasks.
<br>


This task is part of my internship at Prodigy InfoTech, where i implemented a **support vector machine (SVM)** to classify images of cats and dogs.
<br>

**--Project Overview--**
<br>

This project implements an image classification model to distinguish between cats and dogs using a Support Vector Machine (SVM) classifier. Instead of training a Convolutional Neural Network (CNN) from scratch, we leverage MobileNetV2 as a feature extractor to improve efficiency and accuracy.
<br>


**--Tools Used--**
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
<br>

**numpy**: For numerical computations.
<br>
**os** : To navigate through dataset directories.
<br>
**cv2 (OpenCV)** : For image processing (resizing, reading images).
<br>
**pandas**: For data manipulation and preprocessing.
<br>
**tensorflow.keras.applications.MobileNetV2** : Pretrained CNN model for feature extraction.
<br>
**tensorflow.keras.applications.mobilenet_v2.preprocess_input** : To preprocess images before feeding them into MobileNetV2.
<br>
**sklearn.svm.SVC** : Support Vector Machine (SVM) classifier.
<br>
**sklearn.model_selection.train_test_split** : To split the dataset into training and testing sets.
<br>
**sklearn.preprocessing.LabelEncoder** : To encode categorical labels (cats & dogs).
<br>
**sklearn.metrics.accuracy_score** : To measure model accuracy.
<br>
**sklearn.metrics.confusion_matrix** : To analyze classification performance.
<br>
**sklearn.metrics.classification_report**  To generate precision, recall, and F1-score for model evaluation.
<br>
**matplotlib**: A widely used data visualization library.
<br>

**--Implementation Steps--**
<br>

(1)  **Load & Preprocess the Data**
<br>
:) Read images from directories using OpenCV (cv2).
<br>
:) Resize images to 224x224 pixels to match MobileNetV2 input requirements.
<br>
:) Normalize pixel values.
<br>
:) Encode labels ("Cat" → 0, "Dog" → 1).
<br>

(2) **Feature Extraction using MobileNetV2**
<br>
:) Load MobileNetV2 (pre-trained on ImageNet) without the top layer.
<br>
:) Extract deep learning features from images.
<br>

(3) **Train an SVM Classifier**
<br>
:) Use Scikit-Learn's SVM (Support Vector Machine) with a linear kernel.
<br>
:) Train on extracted features.
<br>

**--Results & Visualization--** 
<br>
:) 𝐎𝐯𝐞𝐫𝐚𝐥𝐥 𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲: 90% on test data
<br>
:) 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐑𝐞𝐩𝐨𝐫𝐭 : Produced a detailed classification report:
<br>
 𝐏𝐫𝐞𝐜𝐢𝐬𝐢𝐨𝐧: 0.94 (cats), 0.87 (dogs)
<br>
 𝐑𝐞𝐜𝐚𝐥𝐥: 0.86 (cats), 0.94 (dogs)
<br>
 𝐅𝟏-𝐒𝐜𝐨𝐫𝐞: 0.90 (cats), 0.90 (dogs)
<br>


