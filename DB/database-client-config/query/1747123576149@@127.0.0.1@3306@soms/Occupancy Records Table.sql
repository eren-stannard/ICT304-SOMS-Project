-- Table for occupancy records captured from video feeds
CREATE TABLE OccupancyRecords (
    RecordID INT AUTO_INCREMENT PRIMARY KEY,
    CameraID INT NOT NULL,
    RoomID INT NOT NULL,
    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    OccupancyCount INT,
    FOREIGN KEY (CameraID) REFERENCES Cameras(CameraID),
    FOREIGN KEY (RoomID) REFERENCES Rooms(RoomID)
);