# db_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for Database.

"""


# Libraries used
import mysql.connector
import pandas as pd
import streamlit as st
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
from typing import Any


mydb = mysql.connector.connect(**st.secrets.db_credentials)

mycursor = mydb.cursor()

mycursor.execute("SHOW DATABASES")

for x in mycursor:
  print(x)

mycursor.close()
mydb.close()

#@st.cache_resource
def init_connection() -> PooledMySQLConnection | MySQLConnectionAbstract:
    return mysql.connector.connect(**st.secrets.db_credentials)

#@st.cache_data(ttl=600)
def run_query(query: str) -> Any:
    conn = init_connection()
    with conn.cursor(dictionary=True) as cur:
        cur.execute(query)
        return cur.fetchall()

# Perform query
df = run_query("DESCRIBE occupancyrecords;")

# Print results
st.write(pd.DataFrame(df))