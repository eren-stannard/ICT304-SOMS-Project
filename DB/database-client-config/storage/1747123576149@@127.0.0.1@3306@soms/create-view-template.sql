

CREATE TRIGGER log_insert
AFTER INSERT ON OccupancyRecords
FOR EACH ROW
BEGIN
    INSERT INTO ReadingLog (reading_id, log_time)
    VALUES (NEW.reading_id, NOW());
END 

