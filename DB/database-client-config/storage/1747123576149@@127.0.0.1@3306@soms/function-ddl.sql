
CREATE DEFINER=`root`@`localhost` FUNCTION `GetAverageOccupancy`(cam_id INT) RETURNS float
    DETERMINISTIC
BEGIN
    DECLARE avg_occ FLOAT;
    SELECT AVG(occupancy_count) INTO avg_occ
    FROM OccupancyReadings
    WHERE camera_id = cam_id;
    RETURN avg_occ;
END