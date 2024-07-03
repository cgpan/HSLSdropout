# Codes and Data for *Examining the Algorithmic Fairness in Predicting High School Dropouts*(accepted by EDM2024)
 Chenguang Pan and Zhou Zhang

# 1. Instruction  

We have open-sourced the data and code of our paper *Examining the Algorithmic Fairness in Predicting High School Dropouts* accepted by Educational Data Mining 2024. This project is based on the public version of a nationally representative dataset called *High School Longitudinal Study of 2009* (HSLS:09). You can find the detailed information [here](https://nces.ed.gov/surveys/hsls09/).

  
The raw data can be accessed on NCES's [DataLab](https://nces.ed.gov/datalab/onlinecodebook/session/codebook/c48ab202-0e20-4537-9fbf-96d7d37afd55). Please click the download button on the right side of this webpage and choose the R version (i.e., in the `.rdata` format). The cleaned data is in the `01_data/02_processed_data` folder.  

The codebook and official documents are in the `04_files` folder.  

We welcome any comments, questions, and bug reports on this study. This current study is accepted as a short (less mature) paper rather than a full paper. We are currently working on the extension, and we plan to improve the predictive performance, conduct a more comprehensive examination of the algorithmic fairness in High school dropout prediction, and investigate ways to reduce the predictive bias.  

`Github Copilot`, an AI coding assistant, was used to assist in the development of these coding scripts.

# 2. Running the code  

## 2.1 Our coding environment  
- System: Mac OS Sonoma 14.5
- CPU&GPU: Apple silicon M1 Pro
- Unified RAM: 32GB
- R version: 4.3.0 (2023-04-21) -- "Already Tomorrow"
- Python version: 3.8.17
- Pytorch version: 2.0.1
- Sklearn version: 1.2.2

For Mac users: We noticed that the higher version of Python might cause a conflict between the `sklearn` and `PyTorch,` which sometimes leads to crashes in Jupyter Notebook. Therefore, we switched to Python 3.8.17, and the code ran well. Another tricky thing is that running the neural network model (in `02_NN_models.ipynb`) on an Apple CPU is faster than running on the `mps` (Metal Performance Shaders) when using the Pytorch 2.0.1. 

For Windows users: all those codes should run well on the latest version of R, and Python with the required packages.

## 2.2 Running the scripts  
In the `02_codings` folder:  
- `01_Data_cleaning.r` shows the details about how we clean the raw dataset. The cleaned data were already stored in the `01_data` folder. You directly skip this file if not interested.
- `02_NN_models.ipynb` is to build the neural network models.
- `03_ML_models.ipynb` is about building all the other machine learning models including the logistic regression, random forest, XGBoost, support vector machine.
- `functions.py` contains all the necessary functions used in the scripts above. You can check the details about running the `DAF` functions if you want to apply it to another project. 

 Please email us for feedback, questions, comments, and bug reports. Thank you very much!  
 
 Chenguang Pan and Zhou Zhang
 Email: cp3280@columbia.edu
