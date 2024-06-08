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
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib python3 python3-pip

pip3 install pandas openpyxl psycopg2-binary sqlalchemy

sudo service postgresql start


brew install postgresql

brew services start postgresql

pip3 install pandas openpyxl psycopg2-binary sqlalchemy

DB_NAME="neural_net"
DB_USER=$(whoami)
DB_PASSWORD=""  
DB_HOST="localhost"
DB_PORT="5432"
EXCEL_FILE_PATH="/Users/ssq/Downloads/mfu.xlsx"
TABLE_NAME="mfu"

psql -c "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '${DB_NAME}' AND pid <> pg_backend_pid();"
psql -c "DROP DATABASE IF EXISTS ${DB_NAME};"

createdb ${DB_NAME}
psql -d ${DB_NAME} -c "
CREATE TABLE ${TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    vender VARCHAR(50) NOT NULL,
    functional VARCHAR(50) NOT NULL,
    price INTEGER NOT NULL,
    refueling_cost INTEGER NOT NULL,
    supplie_cost INTEGER NOT NULL,
    repairability VARCHAR(50) NOT NULL,
    parts_support VARCHAR(50) NOT NULL,
    manufacturer VARCHAR(50) NOT NULL,
    efficiency VARCHAR(50) NOT NULL,
    cluster VARCHAR(50) NOT NULL
);
"

echo "
import pandas as pd
from sqlalchemy import create_engine

excel_file = '${EXCEL_FILE_PATH}'
df = pd.read_excel(excel_file)

print('Columns in the Excel file:', df.columns)

df = df.drop(columns=['id'])

engine = create_engine(f'postgresql+psycopg2://${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}')

df.to_sql('${TABLE_NAME}', engine, if_exists='append', index=False)
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
