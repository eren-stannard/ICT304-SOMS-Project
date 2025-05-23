
CREATE DEFINER=`root`@`localhost` PROCEDURE `AddReading`(
    IN p_camera_id INT,
    IN p_timestamp DATETIME,
    IN p_occupancy_count INT
)
BEGIN
    INSERT INTO OccupancyRecords (camera_id, timestamp, occupancy_count)
    VALUES (p_camera_id, p_timestamp, p_occupancy_count);
END