# CLAfICLe

Cross-Lingual Adaptation for In-Context Learning

## Requirements and Setup

### Required Packages

Details such as python and package versions can be found in the generated
[pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock) files.

We recommend using an environment manager such as
[conda](https://docs.conda.io/en/latest/). After setting up your environment
with the correct python version, please proceed with the installation of the
required packages

For [poetry](https://python-poetry.org/) users, getting setup is as easy as
running

```terminal
poetry install
```

We also provide a [requirements.txt](requirements.txt) file for
[pip](https://pypi.org/project/pip/) users who do not wish to use poetry. In
this case, simply run

```terminal
pip install -r requirements.txt
```

This `requirements.txt` file is generated by running the following

```terminal
sh gen_pip_reqs.sh
```

### Checkpoints

We provide the checkpoints for the models used in our paper. These can be
downloaded using the `claficle/run/checkpoints.py` script.

The above script downloads all checkpoints except for the one necessary for our
baseline sandwhich model, which wraps an English `hr_to_lr` MetaICL model with a
translation API. For this checkpoint, please refer to the instructions on the
[MetaICL repo](https://github.com/facebookresearch/MetaICL) for downloading
their `metaicl` model in the `hr_to_lr` setting. Once downloaded, rename this to
`metaicl.pt` and place it in the relevant checkpoints directory.

## Project Organization

```plaintext
    ├── LICENSE
    ├── README.md          <- The top-level README
    ├── data/
    │   ├── interim/       <- Intermediate data that has been transformed.
    │   ├── processed/     <- The final, canonical data sets for modeling.
    │   └── raw/           <- The original, immutable data dump.
    ├── checkpoints/       <- Trained and serialized models.
    ├── notebooks/         <- Jupyter notebooks.
    ├── slurm/             <- Slurm scripts
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    ├── pyproject.toml     <- project metadata, handled by poetry.
    ├── poetry.lock        <- resolving and locking dependencies, handled by poetry.
    ├── requirements.txt   <- for non-poetry users.
    ├── gen_pip_reqs.sh    <- for generating the pip requirements.txt file
    └── claficle/          <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        ├── data/          <- Scripts to download or generate data
        ├── models/        <- Model definitions
        ├── run/           <- scripts to train, evaluate and use models
        ├── conf/          <- config files
        ├── utils.py       <- miscellaneous utils
        └── visualization/ <- Scripts for visualization
```

The project structure is largely based on the
[cookiecutter data-science template](https://github.com/drivendata/cookiecutter-data-science).
This is purposely opinionated so that paths align over collaborators without
having to edit config files. Users may find the
[cookiecutter data-science opinions page](http://drivendata.github.io/cookiecutter-data-science/#opinions),
of relevance

The top level `data/` and `models/` directory are in version control only to
show structure. Their contents will not be committed and are ignored via
`.gitignore`.
