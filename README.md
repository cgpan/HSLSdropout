# Code and Data for *Examining the Algorithmic Fairness in Predicting High School Dropouts*(accepted by EDM2024)
 Chenguang Pan and Zhou Zhang

# 0. Instruction  

We open-source the data and code of our paper *Examining the Algorithmic Fairness in Predicting High School Dropouts* accepted by Educational Data Mining 2024. This project is based on the public version of a nationally representative dataset called *High School Longitudinal Study of 2009* (HSLS:09). You can find the detailed information [here](https://nces.ed.gov/surveys/hsls09/).  

  
The raw data can be accessed on NCES's [DataLab](https://nces.ed.gov/datalab/onlinecodebook/session/codebook/c48ab202-0e20-4537-9fbf-96d7d37afd55). Please click the download button on the right side of this webpage and choose the R version (i.e., in the `.rdata` format).  

The codebook and official documents are in the `04_files` folder.  

We welcome any comments, questions, and bug reports on this study. This current study is accepted as a short (less mature) paper rather than a full paper. We are currently working on the extension of it, and we hope to conduct a more comprehensive examination of the algorithmic fairness in High school dropout prediction.  

`Github Copilot`, an AI coding assistant, was used to assist the development of these coding scripts.

# 1. Running the code  

## 1.1 Our coding environment  
- System: Mac OS Sonoma 14.5
- CPU&GPU: Apple silicon M1 Pro
- Unified RAM: 32GB
- R version: 4.3.0 (2023-04-21) -- "Already Tomorrow"
- Python version: 3.8.17
- Pytorch version: 2.0.1
- Sklearn version: 1.2.2

We noticed that the higher version of Python may cause a conflict between the `sklearn` and `PyTorch,` which sometimes leads to crashes in Jupyter Notebook. Therefore, we switched to Python 3.8.17, and the code ran well. Another tricky thing is running this project on an Apple CPU is faster than running on the `mps` (Metal Performance Shaders) when using the Pytorch 2.0.1. 
