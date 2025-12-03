# Umami-Transformer

## 1. Related Publications
This project is based on or inspired by the following research paper:  
**Fu, B., Fan, M., Yi, J., Du, Y., Tian, H., Yang, T., Cheng, S., & Du, M.** (2025). **Umami-Transformer: A Deep Learning Framework for High-Precision Prediction, Molecular Validation, and Sensory Optimization of Umami Peptides.** *Food Chemistry*. https://doi.org/10.1016/j.foodchem.2025.145905

## 2. License
Umami-Transformer is released under a [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

If you use this work in your research, please cite the original paper using the following BibTeX:

```bibtex
@article{fu2025umami,
  title={Umami-transformer: A deep learning framework for high-precision prediction and experimental validation of umami peptides},
  author={Fu, Baifeng and Fan, Min and Yi, Junjie and Du, Yingxue and Tian, Hong and Yang, Tao and Cheng, Shuzhen and Du, Ming},
  journal={Food Chemistry},
  pages={145905},
  year={2025},
  publisher={Elsevier}
}
```

## 3. Installation

We have tested Umami-Transformer with **Ubuntu 20.04**. It is recommended to use a decently powerful computer for the Server Node to ensure good performance for quickly used.

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
**Note：Due to the special significance of NA and NAN in computer programs, when there are sequences containing NA or NAN, please separate them and manually input these two sequences in the input box**
```
http://www.peptideinnov.com/
```
