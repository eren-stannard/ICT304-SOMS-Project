# __init__.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Append ODS directory to path to access from other SOMS directories.

"""


# Libraries used
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ODS')))