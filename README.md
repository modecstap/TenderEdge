# TenderEdge

TenderEdge is a project aimed at providing a comprehensive platform for predictive modeling and analysis. The repository contains various modules and scripts designed to facilitate the prediction of items using machine learning techniques.

# Installation
### Install python:
```bash
sudo apt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
```
### Install PostgreSQL:
```basg
sudo apt-get install postgresql postgresql-contrib
```

### To install the required dependencies, run:

```python
pip install -r requirements.txt
```

### Push example data from .csv to Postgres
```bash
sudo apt-get install -y postgresql postgresql-contrib python3 python3-pip

pip3 install pandas openpyxl psycopg2-binary sqlalchemy

sudo service postgresql start

DB_NAME="tenderedge_db"
DB_USER="tenderedge_user"
DB_PASSWORD="password"
DB_HOST="localhost"
DB_PORT="5432"
EXCEL_FILE_PATH="/mnt/data/mfu.xlsx"
TABLE_NAME="mfu_data"

sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME};"
sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH ENCRYPTED PASSWORD '${DB_PASSWORD}';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"

echo "
import pandas as pd
from sqlalchemy import create_engine

excel_file = '${EXCEL_FILE_PATH}'
df = pd.read_excel(excel_file)

engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

df.to_sql('${TABLE_NAME}', engine, if_exists='replace', index=False)
" > load_data.py

python3 load_data.py

rm load_data.py

echo "Data has been successfully loaded into the PostgreSQL database."
```

#### to find out the path of the data file(.csv):
```
cd path/to/csv/file
pwd
```

# Usage
### Change config in ./storage/config.py
```python
class DbConfig:
    user = "dbusername"
    password = "dbpassword"
    host = "localhost"
    port = "5432"
    dname = "neural_net"
```

### To run the main script, use the following command:
```python
python main.py
```
# Project Structure

The project is organized into several directories:

- extensions/: Contains extension modules for additional functionalities.
- front/: Frontend components of the project.
- item_predictors/: Scripts for different item prediction algorithms.
- storage/: Modules related to data storage and handling.
- .idea/: Configuration files for the development environment.
- affinityPropogation.py: Script for affinity propagation clustering.
- categoricalEncoder.py: Script for encoding categorical variables.
- neural_network.py: Neural network implementation.
