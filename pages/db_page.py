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
import mysql.connector.types as mysqlt
import pandas as pd
import streamlit as st
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection


@st.cache_resource
def init_connection() -> PooledMySQLConnection | MySQLConnectionAbstract:
    """
    Initialise connection to SOMS MySQL database.
    
    Returns
    -------
    conn : PooledMySQLConnection | MySQLConnectionAbstract
        MySQL database connection.
    """
    
    return mysql.connector.connect(**st.secrets.db_credentials)


@st.cache_data(ttl=600)
def run_query(query: str) -> list[mysqlt.RowType | dict[str, mysqlt.RowItemType]]:
    """
    Query SOMS database with MySQL query.
    
    Parameters
    ----------
    query : str
        SQL query.
    
    Returns
    -------
    rows : list[RowType | dict[str, RowItemType]]
        Result of query.
    """
    
    # Connect to SOMS database
    conn = init_connection()
    
    with conn.cursor(dictionary=True) as cur:
        
        cur.execute(query)
        
        return cur.fetchall()


# Get input query
query = st.text_area("Enter SQL query:")

if query:
    
    # Perform query
    result = run_query(query)

    # Print results
    st.write(pd.DataFrame(result))