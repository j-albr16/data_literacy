# Data-Literacy Project 2025
In our project, we scraped images of politicians from newsoutlets and used face detection and emotion recognition to analyze how different newspapers represent political parties. 


## Analysing Politicians Facial Expressions
This repository contains all scripts/notebook we used for emotion detection, exploratory data analysis and creation of plots.

<div align="center">
  <img src="assets/trump.jpg" width="600" />
</div>

## Usage Instructions
### Getting the data
As we want to prevent commercial usage with the data we gathered, our dataset is not directly available in this repository.

Therefore all of the scripts wont work by default.

If you need access to the datasets, feel free to contact any of the contributiors.

### Using the data
The four main things needed for our scripts are:
1) `newspaper_collection_evaluation_results_20_12_2025.csv` (the dataset containing the results with names, emotions, etc. - **needed for running the scripts in** `/plots`)
2) `stern_sz_evaluation_results_2021to2025_06_01_2026.csv`(the dataset containing the results of a larger timeframe with names, emotions, etc. - **needed for running the scripts in** `/plots`)
2) `politician_image_dataset` (raw image data for processing - **needed for running** `process.py`)
3) `politicians` (reference images of the politicans for face detection - **needed for running** `process.py`)



## Use Pre-Commit for iPyNotebooks
### Installation
- Install the requirements (or `pip install pre-commit`)
- Execute **once** `pre-commit install`
### When Commiting
Assuming you do:

```bash
($ git add nb.ipynb) // optional
$ git commit -m "update nb" nb.ipynb
```

If you changed the .ipynb file you will see:

```bash
nbstripout...............................................................Failed
- hook id: nbstripout
- files were modified by this hook
```

After that `add` and `commit` the `.ipynb` again:
```bash
$ git add nb.ipynb
$ git commit -m "msg" nb.ipynb
```

## (Suggested) Repository Structure
```
.
├── assets
│   └── trump.jpg
├── data                        # example structure of the data folder. Can be adjust via args
│   ├── newspaper_collection_evaluation_results_20_12_2025.csv
│   ├── politician_image_data_set
│   ├── stern_sz_evaluation_results_2021to2025_06_01_2026.csv
│   └── politicians # 
│       ├── afd
│       ├── data.csv
│       ├── fdp
│       ├── gruenen
│       ├── linke
│       ├── spd
│       ├── transnational
│       └── union
├── Makefile                    # shorthands to run scripts
├── models
│   ├── emotion.py              # model to detect facial expression
│   └── recognition.py          # model to label politician faces
├── plots
│   └── exploratory             # plots for exploratory purposes
├── process.py
├── requirements.txt
├── utils.py
└── venv
```
