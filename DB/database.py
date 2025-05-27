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


class Database:
    """SOMS database class for occupancy records."""
    
    def __init__(self, autoconnect: bool = False) -> None:
        """Database constructor."""
        
        self.conn = None
        if autoconnect:
            self.connect_database()
        
        return

    def connect_database(self) -> None:
        """Initialise connection to SOMS MySQL database."""
        
        self.close_connection()
        self.conn = init_connection()
        
        return
    
    def close_connection(self) -> None:
        """Close database connection."""
        
        if self.conn:
            self.conn.close()
            self.conn = None
        
        return
    
    def reset_connection(self) -> None:
        """Reset database connection."""
        
        self.close_connection()
        self.connect_database()
        
        return

    def query_database(self, query: str, values: tuple = (), commit: bool = False) -> list[tuple[str, pd.DataFrame]] | None:
        """
        Perform Database query and return result as pandas DataFrame.
        
        Parameters
        ----------
        query : str
            SQL query to run.
        values : tuple, optional, default=()
            Values to insert into query.
        commit : bool, optional, default=False
            Commit updates to database.
        
        Returns
        -------
        rows_df : list[tuple[str, DataFrame]]
            Result of query.
        """
        
        if self.conn:
            results = run_query(self.conn, query, values, commit)
            
            if results is not None:
                return [(s, pd.DataFrame(r)) for (s, r) in results]
            else:
                return
        
        else:
            st.error("Error: Database not connected")
            return


def init_connection() -> MySQLConnectionAbstract | PooledMySQLConnection:
    """
    Initialise connection to SOMS MySQL database.
    
    Returns
    -------
    conn : Connection to MySQL database
    """
    
    return mysql.connector.connect(**st.secrets.db_credentials)


def run_query(
    conn: MySQLConnectionAbstract | PooledMySQLConnection, query: str, values: tuple = (), commit: bool = False,
) -> list[tuple[str, list[mysqlt.RowType | dict[str, mysqlt.RowItemType]]]] | None:
    """
    Query SOMS database with MySQL query.
    
    Parameters
    ----------
    conn : MySQLConnectionAbstract | PooledMySQLConnection
        MySQL database connection.
    query : str
        SQL query.
    values : tuple, optional, default=()
        Values to insert into query.
    commit : bool, optional, default=False
        Commit updates to database.
    
    Returns
    -------
    results : list[tuple[str, list[RowType | dict[str, RowItemType]]]]
        Result of query.
    """
    
    results = []
    
    try:
        with conn.cursor(dictionary=True) as cur:
        
            cur.execute(query, values)
            results.append((cur.statement, cur.fetchall()))
            
            while cur.nextset():
                results.append((cur.statement, cur.fetchall()))
            
            if commit:
                conn.commit()
        
        return results
    
    except:
        st.warning(f"Error: Failed to perform query:\n>{query}\n\nCheck SQL syntax.")
        return