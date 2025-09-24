import serial
import json

class Measurement:
    def __init__(self, sensor_id, angle, distance, local_position, timestamp):
        self.sensor_id = sensor_id
        self.angle = angle
        self.distance = distance
        self.local_position = local_position
        self.timestamp = timestamp

class SensorReader:
    def __init__(self, port="COM3", baudrate=115200):  # Change port as needed
        self.serial = serial.Serial(port, baudrate, timeout=1)
        
    def get_measurement(self):
        """Returns a Measurement object when detection received, None otherwise"""
        if self.serial.in_waiting:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                data = json.loads(line)
                
                return Measurement(
                    sensor_id=data['sensor_id'],
                    angle=data['angle'],  # Now servo angle relative to home direction (-90 to +90)
                    distance=data['distance'],
                    local_position=data['local_position'],
                    timestamp=data['timestamp']
                )
            except:
                return None
        return None

# Usage example:
if __name__ == "__main__":
    reader = SensorReader("COM3")  # Change to your port
    
    while True:
        measurement = reader.get_measurement()
        if measurement:
            print(f"Sensor {measurement.sensor_id}: {measurement.distance}m at servo angle {measurement.angle}°")