# Umami-Transformer

# 1 Related Publications

[1] 作者名写两到三个 **论文名**.  *发表地* 时间. **[PDF](放PDF的链接)**.


# 2. License

Umami-Transformer is released under a [GPLv3 license] ([https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html)).

If you use Umami-Transformer in an academic work, please cite:

	放一个bibTax的引用链接

# 3. Installation
﻿
We have tested Umami-Transformer with **Ubuntu 20.04** (ROS Noetic). It is recommended to use a decently powerful computer for the Server Node to ensure good performance for quickly used.

## 3.1 Run locally ##

**Note**: Please ensure your system has an NVIDIA GPU and compatible CUDA drivers installed (tested on RTX 4090 with CUDA 12.6), and verify that Anaconda is installed.

1. Create a conda enveriment:
```
conda create -n umima_test python==3.9
conda activate umami_test
```
2. Install Python Packages
```
pip install pandas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install scikit-learn
pip install tqdm
```
3. Clone the source repo into your workspace folder
```
cd ~/your_workspace
git clone https://github.com/peptideinnov11/Umami-Transformer.git
```
4. Preprocess Test Data for Format Consistency
Ensure your data format strictly adheres to the schema defined in 2_peptide.csv
```
cd Umami-Transformer
python make_feature.py
```
5. Retrieve Prediction Results
If you retained the default output filename from the previous step, proceed directly. Otherwise, modify the corresponding path variables in the Python script.
```
python make_feature.py
```
The inference results can be found in result/result.csv.

## 3.2 Run via the web interface ##
```
http://www.peptideinnov.com/
```
