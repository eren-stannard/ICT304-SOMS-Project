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
import plotly.express as px
import streamlit as st
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection

# FIles used
from DB import database
from DB.database import Database


def main() -> None:
    """Main entry point."""
    
    # Initialise database and connection
    if 'database' not in st.session_state:
        st.session_state.database = load_database()
    
    db = st.session_state.database
    
    if db.conn is None or not db.conn.is_connected():
        db.reset_connection()

    # Get input query
    query = st.text_area(
        label=":primary[:material/database_search:] Enter MySQL Query:",
        height=220,
        help="Query the SOMS database using MySQL statements",
        placeholder="DESCRIBE occupancyrecords;\nSELECT * FROM occupancyrecords;",
    )
    autocommit = st.checkbox(label="Autocommit query", value=False)
    
    # Perform query and display results
    if query:
        results = db.query_database(query, commit=autocommit)
        if results is not None:
            for statement, result in results:
                st.write(statement)
                st.dataframe(result)
                if not result.empty:
                    st.plotly_chart(px.line(result, x='Timestamp', y='OccupancyCount', title="Occupancy Records Over Time"))
    
    return


@st.cache_resource(ttl=600)
def init_connection() -> MySQLConnectionAbstract | PooledMySQLConnection:
    return database.init_connection()

@st.cache_resource(ttl=600)
def load_database() -> Database:
    return Database(True)

@st.cache_data(ttl=600)
def run_query(query: str, commit: bool = False) -> None:
    database.run_query(st.session_state.database.conn, query, commit=commit)
    return


if __name__ == '__main__':
    main()