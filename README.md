fraud_detection
==============================



Instructions
------------
1. Clone the repo.
1. Run `make dirs` to create the missing parts of the directory structure described below.
1. *Optional:* Run `make virtualenv` to create a python virtual environment. Skip if using conda or some other env manager.
    1. Run `source env/bin/activate` to activate the virtualenv.
1. Run `make requirements` to install required python packages.
1. Put the raw data in `data/raw`.
1. To save the raw data to the DVC cache, run `dvc commit raw_data.dvc`
1. Edit the code files to your heart's desire.
1. Process your data, train and evaluate your model using `dvc repro eval.dvc` or `make reproduce`
1. When you're happy with the result, commit files (including .dvc files) to git.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── raw_data.dvc       <- Keeps the raw data versioned.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── dvc.lock           <- The version definition of each dependency, stage, and output from the 
    │                         data pipeline.
    ├── dvc.yaml           <- Defining the data pipeline stages, dependencies, and outputs.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                 <- Generated graphics and figures to be used in reporting
    │   └── metrics.txt             <- Relevant metrics after evaluating the model.
    │   └── training_metrics.txt    <- Relevant metrics from training the model.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── __init__.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── __init__.py
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------
Data

PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world.

    step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
    type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
    amount - amount of the transaction in local currency.
    nameOrig - customer who started the transaction
    oldbalanceOrg - initial balance before the transaction
    newbalanceOrig - new balance after the transaction
    nameDest - customer who is the recipient of the transaction
    oldbalanceDest - initial balance recipient before the transaction. Note that there is no information for customers that start with M (Merchants).
    newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
    isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
    isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction


    Cash in
The process by which a customer credits his account with cash. This is usually via an agent who takes the cash and credits the
customer’s mobile money account.
Cash out
The process by which a customer deducts cash from his mobile money account. This is usually via an agent who gives the customer
cash in exchange for a transfer from the customer’s mobile money account.

binary classification problem - i.e. our target variable is a binary attribute (Is the user making the click fraudlent or not?) and our goal is to classify users into "fraudlent" or "not fraudlent" as well as possible.