-- Table for security camera devices
CREATE TABLE Cameras (
    CameraID INT AUTO_INCREMENT PRIMARY KEY,
    RoomID INT NOT NULL,
    IPAddress VARCHAR(50),
    Status ENUM('Active', 'Inactive') DEFAULT 'Active',
    InstalledDate DATE,
    Resolution VARCHAR(50),
    FieldOfView VARCHAR(100),
    FOREIGN KEY (RoomID) REFERENCES Rooms(RoomID)
);