-- Table for user roles and system users
CREATE TABLE Roles (
    RoleID INT AUTO_INCREMENT PRIMARY KEY,
    RoleName VARCHAR(50) UNIQUE
);

CREATE TABLE Users (
    UserID INT AUTO_INCREMENT PRIMARY KEY,
    FullName VARCHAR(100),
    Email VARCHAR(100) UNIQUE,
    PasswordHash VARCHAR(255),
    RoleID INT,
    IsActive BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (RoleID) REFERENCES Roles(RoleID)
);