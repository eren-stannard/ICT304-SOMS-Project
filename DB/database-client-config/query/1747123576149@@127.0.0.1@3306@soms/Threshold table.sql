-- Table for alerts when occupancy exceeds limit
CREATE TABLE Thresholds (
    ThresholdID INT AUTO_INCREMENT PRIMARY KEY,
    RoomID INT,
    MaxOccupancy INT,
    AlertEnabled BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (RoomID) REFERENCES Rooms(RoomID)
);