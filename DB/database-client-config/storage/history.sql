/* 2025-05-13 16:08:49 [2 ms] */ 
DELETE FROM `rooms` WHERE `RoomID` IN (1,2,3);
/* 2025-05-13 16:10:45 [0 ms] */ 
USE soms;
/* 2025-05-13 16:22:13 [3 ms] */ 
CREATE PROCEDURE AddReading(
    IN p_camera_id INT,
    IN p_timestamp DATETIME,
    IN p_occupancy_count INT
)
BEGIN
    INSERT INTO OccupancyReadings (camera_id, timestamp, occupancy_count)
    VALUES (p_camera_id, p_timestamp, p_occupancy_count);
END;
/* 2025-05-13 16:22:31 [2 ms] */ 
CREATE FUNCTION GetAverageOccupancy(cam_id INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE avg_occ FLOAT;
    SELECT AVG(occupancy_count) INTO avg_occ
    FROM OccupancyReadings
    WHERE camera_id = cam_id;
    RETURN avg_occ;
END;
/* 2025-05-16 09:06:02 [18 ms] */ 
CREATE PROCEDURE AddReading(
    IN p_camera_id INT,
    IN p_timestamp DATETIME,
    IN p_occupancy_count INT
)
BEGIN
    INSERT INTO OccupancyReadings (camera_id, timestamp, occupancy_count)
    VALUES (p_camera_id, p_timestamp, p_occupancy_count);
END;
/* 2025-05-16 09:09:27 [5 ms] */ 
CREATE PROCEDURE AddReading(
    IN p_camera_id INT,
    IN p_timestamp DATETIME,
    IN p_occupancy_count INT
)
BEGIN
    INSERT INTO OccupancyRecords (camera_id, timestamp, occupancy_count)
    VALUES (p_camera_id, p_timestamp, p_occupancy_count);
END;
/* 2025-05-16 09:12:22 [9 ms] */ 
CREATE FUNCTION GetAverageOccupancy(cam_id INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    DECLARE avg_occ FLOAT;
    SELECT AVG(occupancy_count) INTO avg_occ
    FROM OccupancyReadings
    WHERE camera_id = cam_id;
    RETURN avg_occ;
END;
