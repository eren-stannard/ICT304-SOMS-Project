-- Table for rooms
CREATE TABLE Rooms (
    RoomID INT AUTO_INCREMENT PRIMARY KEY,
    BuildingID INT,
    RoomName VARCHAR(100) NOT NULL,
    Capacity INT NOT NULL,
    Floor INT,
    FOREIGN KEY (BuildingID) REFERENCES Buildings(BuildingID)
);