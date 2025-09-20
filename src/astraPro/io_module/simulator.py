"""
Realistic movement simulator for objects in a 3m x 3m field.
Generates JSON measurements only (not tracks).
"""

# ===========================================
# SCENARIO CONFIGURATION - CHANGE IN config.yaml
# ===========================================
DEFAULT_SCENARIO = "all"  # Fallback if config.yaml is not available

import time
import math
import random
import queue
import threading
import yaml
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .serial_reader import Measurement


@dataclass
class ObjectState:
    """Represents the current state of a moving object."""
    x: float  # meters
    y: float  # meters
    vx: float  # velocity x (m/s)
    vy: float  # velocity y (m/s)
    ax: float = 0.0  # acceleration x (m/s²)
    ay: float = 0.0  # acceleration y (m/s²)


@dataclass
class SensorConfig:
    """Configuration for a rotating sensor matching hardware."""
    id: str  # sensor ID (e.g., "S_1")
    x: float  # position x (meters)
    y: float  # position y (meters)
    max_range: float  # maximum detection range (meters)
    min_range: float  # minimum detection range (meters)
    field_of_view: float  # angular field of view in degrees
    servo_speed: float  # degrees per cycle
    home_direction: float = 0.0  # home direction angle (degrees)
    current_angle: float = 0.0  # current servo angle
    last_update_time: float = 0.0  # last servo movement time


class MovementPattern:
    """Base class for movement patterns."""
    
    def update_object(self, obj: ObjectState, dt: float, field_bounds: Tuple[float, float]) -> None:
        """Update object state based on pattern."""
        raise NotImplementedError


class LinearMovement(MovementPattern):
    """Linear movement with momentum and boundary bouncing."""
    
    def __init__(self, max_speed: float = 2.0, bounce: bool = True):
        self.max_speed = max_speed
        self.bounce = bounce
    
    def update_object(self, obj: ObjectState, dt: float, field_bounds: Tuple[float, float]) -> None:
        width, height = field_bounds
        
        # Apply acceleration (random small changes)
        if random.random() < 0.1:  # 10% chance to change direction slightly
            obj.ax += random.uniform(-1.0, 1.0)
            obj.ay += random.uniform(-1.0, 1.0)
        
        # Apply friction/damping to acceleration
        obj.ax *= 0.9
        obj.ay *= 0.9
        
        # Update velocity with acceleration
        obj.vx += obj.ax * dt
        obj.vy += obj.ay * dt
        
        # Limit speed
        speed = math.sqrt(obj.vx**2 + obj.vy**2)
        if speed > self.max_speed:
            obj.vx = (obj.vx / speed) * self.max_speed
            obj.vy = (obj.vy / speed) * self.max_speed
        
        # Update position
        obj.x += obj.vx * dt
        obj.y += obj.vy * dt
        
        # Handle boundaries
        if self.bounce:
            if obj.x <= -width/2 or obj.x >= width/2:
                obj.vx = -obj.vx
                obj.x = max(-width/2, min(width/2, obj.x))
            if obj.y <= -height/2 or obj.y >= height/2:
                obj.vy = -obj.vy
                obj.y = max(-height/2, min(height/2, obj.y))
        else:
            # Keep within bounds
            obj.x = max(-width/2, min(width/2, obj.x))
            obj.y = max(-height/2, min(height/2, obj.y))


class CornerToCornerWalk(MovementPattern):
    """Slow steady walk from one corner to another, stops at boundary."""
    
    def __init__(self, start_corner: str = "SW", end_corner: str = "NE", speed: float = 0.5):
        """
        Initialize corner-to-corner walk.
        
        Args:
            start_corner: Starting corner ("SW", "SE", "NW", "NE")
            end_corner: Ending corner ("SW", "SE", "NW", "NE") 
            speed: Walking speed in m/s
        """
        self.speed = speed
        self.completed = False
        
        # Define corner positions for 3x3m field (relative to field center)
        corners = {
            "SW": (-1.5, -1.5),  # Southwest corner of 3x3m field
            "SE": (1.5, -1.5),   # Southeast corner of 3x3m field
            "NW": (-1.5, 1.5),   # Northwest corner of 3x3m field
            "NE": (1.5, 1.5)     # Northeast corner of 3x3m field
        }
        
        self.start_corner = corners[start_corner]
        self.end_corner = corners[end_corner]
        
        # Calculate direction vector
        dx = self.end_corner[0] - self.start_corner[0]
        dy = self.end_corner[1] - self.start_corner[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            self.direction_x = dx / distance
            self.direction_y = dy / distance
        else:
            self.direction_x = 0
            self.direction_y = 0
            self.completed = True
    
    def update_object(self, obj: ObjectState, dt: float, field_bounds: Tuple[float, float]) -> None:
        if self.completed:
            # Stop moving
            obj.vx = 0
            obj.vy = 0
            obj.ax = 0
            obj.ay = 0
            return
        
        width, height = field_bounds
        
        # Set constant velocity toward target
        obj.vx = self.direction_x * self.speed
        obj.vy = self.direction_y * self.speed
        obj.ax = 0
        obj.ay = 0
        
        # Update position
        new_x = obj.x + obj.vx * dt
        new_y = obj.y + obj.vy * dt
        
        # Check if we've reached far outside the sensor range (stop simulation)
        max_distance = 5.0  # Stop when 5m away from center
        distance_from_center = math.sqrt(new_x**2 + new_y**2)
        
        if distance_from_center > max_distance:
            # Stop simulation when very far away
            self.completed = True
            obj.vx = 0
            obj.vy = 0
            obj.ax = 0
            obj.ay = 0
            # Keep current position
        else:
            # Continue moving - go THROUGH the field and outside
            obj.x = new_x
            obj.y = new_y


class CircularMovement(MovementPattern):
    """Circular or oval movement pattern."""
    
    def __init__(self, radius: float = 1.0, angular_speed: float = 1.0, center: Tuple[float, float] = (0, 0)):
        self.radius = radius
        self.angular_speed = angular_speed
        self.center_x, self.center_y = center
        self.angle = 0.0
    
    def update_object(self, obj: ObjectState, dt: float, field_bounds: Tuple[float, float]) -> None:
        self.angle += self.angular_speed * dt
        
        # Calculate circular position
        obj.x = self.center_x + self.radius * math.cos(self.angle)
        obj.y = self.center_y + self.radius * math.sin(self.angle)
        
        # Calculate velocity for smooth motion
        obj.vx = -self.radius * self.angular_speed * math.sin(self.angle)
        obj.vy = self.radius * self.angular_speed * math.cos(self.angle)


class RandomWalk(MovementPattern):
    """Random walk with momentum."""
    
    def __init__(self, step_size: float = 0.1, momentum: float = 0.8):
        self.step_size = step_size
        self.momentum = momentum
    
    def update_object(self, obj: ObjectState, dt: float, field_bounds: Tuple[float, float]) -> None:
        width, height = field_bounds
        
        # Add random component to velocity
        random_vx = random.uniform(-self.step_size, self.step_size)
        random_vy = random.uniform(-self.step_size, self.step_size)
        
        # Apply momentum
        obj.vx = obj.vx * self.momentum + random_vx * (1 - self.momentum)
        obj.vy = obj.vy * self.momentum + random_vy * (1 - self.momentum)
        
        # Update position
        obj.x += obj.vx * dt
        obj.y += obj.vy * dt
        
        # Bounce off walls
        if obj.x <= -width/2 or obj.x >= width/2:
            obj.vx = -obj.vx
            obj.x = max(-width/2, min(width/2, obj.x))
        if obj.y <= -height/2 or obj.y >= height/2:
            obj.vy = -obj.vy
            obj.y = max(-height/2, min(height/2, obj.y))


class RealisticMovementSimulator:
    """Realistic movement simulator for objects with rotating sensors matching hardware."""
    
    def __init__(self, field_size: Tuple[float, float] = (3.0, 3.0)):
        self.field_width, self.field_height = field_size
        
        # 3x3m field sensors at corners with optimal coverage directions
        self.sensors = [
            SensorConfig(1, -1.5, -1.5, 2.5, 0.02, 20.0, 30.0, 45.0),    # Southwest corner, points toward NE
            SensorConfig(2, 1.5, -1.5, 2.5, 0.02, 20.0, 30.0, 135.0),    # Southeast corner, points toward NW  
            SensorConfig(3, 1.5, 1.5, 2.5, 0.02, 20.0, 30.0, -135.0),    # Northeast corner, points toward SW
            SensorConfig(4, -1.5, 1.5, 2.5, 0.02, 20.0, 30.0, -45.0),    # Northwest corner, points toward SE
        ]
        
        # Initialize sensor angles to their home directions
        for sensor in self.sensors:
            sensor.current_angle = sensor.home_direction
        
        self.objects = []
        self.movement_patterns = []
        self.running = False
        self.thread = None
        
        # Hardware timing parameters
        self.servo_movement_time = 0.05  # 50ms in seconds
        self.sonar_update_time = 0.1     # 100ms in seconds
        self.last_measurement_time = time.time()
        
        # Noise parameters
        self.distance_noise = 0.01  # 1cm std dev (more realistic)
        self.angle_noise = 1.0      # 1 degree std dev
    
    def add_object(self, initial_x: float = 0.0, initial_y: float = 0.0, 
                   pattern: Optional[MovementPattern] = None) -> int:
        """Add an object to the simulation."""
        if pattern is None:
            pattern = LinearMovement()
        
        obj = ObjectState(
            x=initial_x, y=initial_y,
            vx=random.uniform(-1.0, 1.0),
            vy=random.uniform(-1.0, 1.0)
        )
        
        self.objects.append(obj)
        self.movement_patterns.append(pattern)
        return len(self.objects) - 1
    
    def update_sensor_angles(self, current_time: float):
        """Update sensor angles based on servo rotation speed."""
        for sensor in self.sensors:
            # Calculate time since last update
            if sensor.last_update_time == 0.0:
                sensor.last_update_time = current_time
                continue
                
            time_delta = current_time - sensor.last_update_time
            
            # Only update if enough time has passed (servo movement timing)
            if time_delta >= self.servo_movement_time:
                # Update angle based on servo speed
                angle_increment = sensor.servo_speed * (time_delta / self.servo_movement_time)
                sensor.current_angle += angle_increment
                
                # Keep angle in reasonable range (simulate 180-degree sweep around home direction)
                max_angle = sensor.home_direction + 90
                min_angle = sensor.home_direction - 90
                
                if sensor.current_angle > max_angle:
                    sensor.current_angle = max_angle
                    sensor.servo_speed = -abs(sensor.servo_speed)  # Reverse direction
                elif sensor.current_angle < min_angle:
                    sensor.current_angle = min_angle
                    sensor.servo_speed = abs(sensor.servo_speed)   # Reverse direction
                
                sensor.last_update_time = current_time

    def calculate_sensor_measurement(self, sensor: SensorConfig, obj: ObjectState, current_time: float) -> Optional[Measurement]:
        """Calculate what a rotating sensor would measure for a given object."""
        # Calculate relative position
        dx = obj.x - sensor.x
        dy = obj.y - sensor.y
        
        # Calculate distance
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if within sensor range
        if distance > sensor.max_range or distance < sensor.min_range:
            return None
        
        # Calculate angle from sensor to object (in world coordinates)
        obj_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize angle to [-180, 180]
        while obj_angle > 180:
            obj_angle -= 360
        while obj_angle < -180:
            obj_angle += 360
        
        # Calculate angle difference between sensor pointing direction and object
        angle_diff = abs(obj_angle - sensor.current_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Check if object is within sensor's field of view
        if angle_diff > sensor.field_of_view / 2:
            return None
        
        # Simulate measurement delay - only generate measurements at sonar update intervals
        if current_time - self.last_measurement_time < self.sonar_update_time:
            return None
        
        # Add realistic noise
        noisy_distance = distance + random.gauss(0, self.distance_noise)
        noisy_obj_angle = obj_angle + random.gauss(0, self.angle_noise)
        
        # Ensure realistic bounds
        noisy_distance = max(sensor.min_range, min(sensor.max_range, noisy_distance))
        
        return Measurement(
            sensor_id=sensor.id,
            angle=noisy_obj_angle,  # Angle from sensor to object, not servo angle
            distance=noisy_distance,
            local_position=[0, 0],  # Not used
            timestamp=current_time
        )
    
    def update_simulation(self, dt: float):
        """Update all objects in the simulation."""
        field_bounds = (self.field_width, self.field_height)
        
        for obj, pattern in zip(self.objects, self.movement_patterns):
            pattern.update_object(obj, dt, field_bounds)
    
    def get_all_measurements(self, current_time: float) -> List[Measurement]:
        """Get measurements from all rotating sensors for all objects."""
        # Update sensor angles based on servo rotation
        self.update_sensor_angles(current_time)
        
        measurements = []
        measurement_generated = False
        
        for obj in self.objects:
            for sensor in self.sensors:
                measurement = self.calculate_sensor_measurement(sensor, obj, current_time)
                if measurement:
                    measurements.append(measurement)
                    measurement_generated = True
        
        # Update last measurement time if any measurements were generated
        if measurement_generated:
            self.last_measurement_time = current_time
        
        return measurements
    
    def get_ground_truth_objects(self) -> List[ObjectState]:
        """Get current ground truth object states."""
        return self.objects.copy()
    
    def simulate_measurements(self, raw_queue: queue.Queue, update_rate: float = 20.0):
        """Generate simulated measurements with rotating sensors and put them in raw queue."""
        dt = 1.0 / update_rate
        
        while self.running:
            start_time = time.time()
            
            # Update object positions
            self.update_simulation(dt)
            
            # Generate measurements from rotating sensors
            measurements = self.get_all_measurements(start_time)
            
            # Add measurements to queue
            for measurement in measurements:
                raw_queue.put(measurement)
            
            # Debug output (optional)
            if measurements:
                active_sensors = [m.sensor_id for m in measurements]
                print(f"Generated {len(measurements)} measurements from sensors: {active_sensors}")
            
            # Maintain timing
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    def start(self, raw_queue: queue.Queue, update_rate: float = 20.0):
        """Start the simulation in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self.simulate_measurements,
            args=(raw_queue, update_rate),
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


def _load_scenario_from_config() -> str:
    """Load scenario setting from config.yaml."""
    try:
        # Try to find config.yaml - look up from current directory
        config_paths = [
            "config.yaml",
            "../config.yaml", 
            "../../config.yaml",
            "../../../config.yaml"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    scenario = config.get('simulation', {}).get('scenario', DEFAULT_SCENARIO)
                    print(f"Loaded scenario '{scenario}' from {config_path}")
                    return scenario
        
        print(f"config.yaml not found, using default scenario: {DEFAULT_SCENARIO}")
        return DEFAULT_SCENARIO
        
    except Exception as e:
        print(f"Error loading config.yaml: {e}, using default scenario: {DEFAULT_SCENARIO}")
        return DEFAULT_SCENARIO


def create_default_simulation(scenario: str = None) -> RealisticMovementSimulator:
    """
    Create a simulation with configurable scenarios for 3x3m field.
    
    Args:
        scenario: Which scenario to run
            - "all": All three patterns (default)
            - "diagonal": Only corner-to-corner diagonal walk
            - "circle": Only circular movement in center
            - "random": Only random walk
            - "diagonal_circle": Diagonal walk + circular movement
            - "diagonal_random": Diagonal walk + random walk
            - "circle_random": Circular movement + random walk
    """
    sim = RealisticMovementSimulator()
    
    # Load scenario from config.yaml if none provided
    if scenario is None:
        scenario = _load_scenario_from_config()
    
    if scenario in ["all", "diagonal", "diagonal_circle", "diagonal_random"]:
        sim.add_object(-1.0, -1.0, CornerToCornerWalk(start_corner="SW", end_corner="NE", speed=0.5))
        
    if scenario in ["all", "circle", "diagonal_circle", "circle_random"]:
        sim.add_object(0.0, 0.0, CircularMovement(radius=0.8, angular_speed=0.5))
        
    if scenario in ["all", "random", "diagonal_random", "circle_random"]:
        sim.add_object(0.5, -0.5, RandomWalk(step_size=0.2, momentum=0.7))
    
    return sim


def simulate_measurements(raw_queue: queue.Queue):
    """Simple replacement for the old simulate_measurements function."""
    simulator = create_default_simulation()
    simulator.start(raw_queue)
    
    # Keep the thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        simulator.stop()