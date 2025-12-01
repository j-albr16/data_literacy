# Data-Literacy Project 2025
## Analysing Politicians Facial Expressions
![Trump](assets/trump.jpg)

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
│   ├── bundestag_members_with_paths.csv
│   ├── out.csv
│   ├── politician_data_set
│   │   ├── images
│   │   └── politicians.csv
│   ├── politician_data_set.zip
│   └── politicians
│       ├── afd
│       ├── data.csv
│       ├── ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl
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
│   ├── analysis                # plots for analysis
│   └── exploratory             # plots for exploratory purposes
├── process.py
├── requirements.txt
│   │   └── recognition.cpython-313.pyc
├── utils.py
└── venv

28 directories, 57 files
```
