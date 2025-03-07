# Handwritten Chinese character recognition based on MobileNetV3


## 1. Introduction
This project used the lightweight neural network MobileNetV3 to achieve efficient and accurate handwritten Chinese character recognition. MobileNetV3 significantly reduces the model complexity and computation while maintaining high accuracy, which provides basic support for subsequent processing.


## 2. Dependencies
>python >= 3.8  
>torch >= 1.10.0  
>torchvision >= 0.11.0  
>opencv-python >= 4.8.0.76


## 3. HWDB Dataset
### 3.1. Brief Description
[HWDB](https://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) is a handwritten Chinese character dataset. The dataset comes from the Institute of Automation, Chinese Academy of Sciences. There are three versions of HWDB1.0, HWDB1.1 and HWDB1.2, respectively. In this experiment, we used the HWDB1.1 dataset with a total of 1,176,000 images. The dataset is handwritten by 300 people, which contains 171 Arabic numerals and special symbols, and 3755 GB2312-80 level-1 Chinese characters. Both them, the training set contains words written by 240 writers with 119,516 image, and the test set contains words written by another 60 writers with a total of 29,859 images.

### 3.2. Download
[Official website](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html),  [Training set](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip),  [Test set](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip)

### 3.3. Parsing
It is worth noting that the downloaded dataset is not an image format, but a custom gnt file type. Therefore, the gnt file needs to be converted to all the png images in the corresponding label directory.

$\circ$ Step 1

Extract the downloaded zip file into a gnt file. To be notieced that, HWDB1.1trn_gnt.zip unzipped is alz file, so it needs to unzipped alz file again.

Unzip the alz file：
Windows：need to download software https://alzip.en.softonic.com/download)

Linux：unalz HWDB1.1trn_gnt.alz

$\circ$ Step 2:

Convert the gnt file to all png images in the corresponding label directory. Modify the path in process_gnt.py to run the program.

## 4. Model 

Compared to its MobileNetV1 and MobileNetV2, MobileNetV3 has three improvements: First, it adjusts the input and output layers of the network, which reduces the delay of backpropagation and sustains the high-dimensional feature space. Second, instead of Sigmoid, a new activation function called H − swish is embedded in MobileNetV3 to decrease computational consumption. Lastly, the SE block is integrated after the depth-wise separable convolution in the inverse residual module, and the bottleneck structure is stimulated to better extract image features synchronously.
