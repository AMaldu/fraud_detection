fraud_detection
==============================


--------
Dataset context

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





Instructions
------------

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AMaldu/fraud_detection
   cd fraud-detection


2. **Create virtual environment**: To install the dependencies and activate the virtual environment, run the following commands:
    ```bash
    pipenv --3.10
    pipenv shell

3. **Install the dependencies**:
    ```bash
    pipenv install -r requirements.txt

4. **Running the project**:
    ```bash
    docker compose up


How it works
------------

**Notebooks**: this folder contains the part with the EDA
- basic EDA
- univariate analysis
- bivariate analysis
- multivariate analysis
- individual analysis of some features of interest

**Experiments**: This folder contains the part with the experimentation using MLFlow
1. Run the MLFlow server:
    ```bash
    cd experiments
    mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
    ```
2. Now you can play with the the 'random_forest_reduced.ipynb' file and see the results on the UI of MLFlow.

   **Note**: The original dataset is very large, so a version of the experiments with a reduced dataset is also provided.

**Orchestration**: This folder contains the scripts used for orchestration in Prefect Server
1. Run the Prefect server:
    ```bash
    prefect server start
    ```
2. Set the API URL for the server:
    ```bash
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
    ```
3. Run the `orchestrate.py` file or check the pipelines that are already inside Prefect.

**Deployment**: This folder contains the scripts used for deploying the preprocessor and the model inside a Docker container.
1. Run the Dockerfile
    ```bash
    docker build -t fraud_detection_app -f src/Dockerfile .
    docker run -p 9696:9696 fraud_detection_app
    ```

**Monitorization**: This folder contains the notebooks for monitoring the model with Grafana

1. Run
    ```bash
    docker compose up --build
    ```
2. Run the jupyter notebooks

3. Run the 'metrics_calculation.py file and acces the Adminer database: http://localhost:8080/

4. Open Grafana to see the monitorization metrics and alerts: http://localhost:3000/
