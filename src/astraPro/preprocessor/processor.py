"""
Preprocessor that pops measurements from raw queue and pushes to processed queue.
"""

import queue
import threading
import yaml
from typing import Dict, Any
from ..io_module.serial_reader import Measurement
from .transforms import measurement_to_world


class ProcessedMeasurement:
    """Measurement with world coordinates added."""
    
    def __init__(self, original: Measurement, x_world: float, y_world: float):
        self.sensor_id = original.sensor_id
        self.angle = original.angle
        self.distance = original.distance
        self.local_position = original.local_position
        self.timestamp = original.timestamp
        self.x_world = x_world
        self.y_world = y_world


def preprocess_measurements(raw_queue: queue.Queue, processed_queue: queue.Queue, 
                          config_path: str = "config.yaml", debug_print: bool = False):
    """
    Pop measurements from raw queue, transform them, push to processed queue.
    
    Args:
        raw_queue: Queue containing raw Measurement objects
        processed_queue: Queue to put ProcessedMeasurement objects
        config_path: Path to config file
        debug_print: If True, print processed measurements to terminal
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sensors = {sensor['id']: sensor for sensor in config['sensors']}
    field_config = config['field']
    
    while True:
        try:
            # Pop from raw queue (blocking)
            measurement = raw_queue.get(timeout=0.1)
            
            # Get sensor config
            sensor_config = sensors.get(measurement.sensor_id)
            if not sensor_config:
                continue
            
            # Transform to world coordinates
            x_world, y_world = measurement_to_world(
                measurement.distance,
                measurement.angle,
                sensor_config,
                field_config
            )
            
            # Create processed measurement
            processed = ProcessedMeasurement(measurement, x_world, y_world)
            
            # Print if debug mode enabled
            if debug_print:
                print(f"Processed: Sensor {processed.sensor_id} - "
                      f"World({processed.x_world:.2f}, {processed.y_world:.2f})")
            
            # Push to processed queue
            processed_queue.put(processed)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Preprocessing error: {e}")


def start_preprocessor(raw_queue: queue.Queue, processed_queue: queue.Queue, 
                      config_path: str = "config.yaml", debug_print: bool = False) -> threading.Thread:
    """
    Start preprocessor in background thread.
    
    Args:
        debug_print: If True, print processed measurements to terminal
    
    Returns:
        The thread running the preprocessor
    """
    thread = threading.Thread(
        target=preprocess_measurements,
        args=(raw_queue, processed_queue, config_path, debug_print),
        daemon=True
    )
    thread.start()
    return thread