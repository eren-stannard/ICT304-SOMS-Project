-- Create database
CREATE DATABASE soms;
USE soms;

-- Table for buildings or monitored areas
CREATE TABLE Buildings (
    BuildingID INT AUTO_INCREMENT PRIMARY KEY,
    BuildingName VARCHAR(100) NOT NULL,
    Location VARCHAR(255)
);
