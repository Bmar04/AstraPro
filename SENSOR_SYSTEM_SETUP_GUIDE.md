# AstraPro Sensor System Setup Guide

## Overview

This document provides comprehensive instructions for setting up the AstraPro multi-sensor tracking system to match your simulator configuration. The system uses 4 ultrasonic sensors positioned at field corners with rotating servo mechanisms for 360-degree coverage.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hardware Requirements](#hardware-requirements)
3. [Field Setup](#field-setup)
4. [Sensor Positioning](#sensor-positioning)
5. [Communication Protocol](#communication-protocol)
6. [Configuration Files](#configuration-files)
7. [Coordinate System](#coordinate-system)
8. [Message Format](#message-format)
9. [Hardware Configuration](#hardware-configuration)
10. [Testing and Validation](#testing-and-validation)
11. [Troubleshooting](#troubleshooting)

## System Architecture

The AstraPro system consists of:
- **4 Ultrasonic Sensors**: HC-SR04 or equivalent with ~2.5m range
- **4 Servo Motors**: For sensor rotation (180° sweep range)
- **Microcontroller**: Arduino/ESP32 for each sensor unit
- **Communication**: Serial communication (115200 baud)
- **Central Processing**: Python-based fusion and tracking system

### Data Flow
```
Physical Sensors → Serial Messages → Raw Queue → Preprocessor → Processed Queue → Triangulation Engine → Tracking System
```

## Hardware Requirements

### Per Sensor Unit
- **Ultrasonic Sensor**: HC-SR04 or equivalent
  - Range: 0.02m to 2.5m
  - Accuracy: ±2cm
  - Field of View: ~20° (hardware limitation)
  
- **Servo Motor**: 
  - Type: Standard hobby servo (SG90 or similar)
  - Range: 180° rotation
  - Speed: 180°/second (configurable)
  - Torque: Sufficient for sensor weight
  
- **Microcontroller**: Arduino Uno/ESP32
  - Serial communication capability
  - Sufficient pins for sensor + servo control
  - Real-time capabilities for precise timing

### Central System
- **Computer**: Running Python 3.8+
- **Serial Interface**: USB-to-Serial adapter or direct connection
- **Processing Power**: Sufficient for real-time data fusion (typical desktop/laptop)

## Field Setup

### Field Dimensions
```yaml
# 3m × 3m field with origin at center
field:
  width: 3.0   # meters
  height: 3.0  # meters
  center:
    x: 0.0     # World coordinate origin
    y: 0.0     # World coordinate origin
```

### Coordinate System
- **Origin**: Center of the 3×3m field
- **X-Axis**: Positive direction points East
- **Y-Axis**: Positive direction points North
- **Z-Axis**: Height above ground (sensors at 0.5m)

```
     Y (North)
     ↑
(-1.5,1.5) ────────── (1.5,1.5)
     │       3m        │
     │                 │
     │        ●        │ → X (East)
     │     (0,0)       │
     │                 │
     │                 │
(-1.5,-1.5) ────────── (1.5,-1.5)
            3m
```

## Sensor Positioning

### Physical Positions
Each sensor must be positioned **exactly** at the specified coordinates:

| Sensor | Position | Corner | Home Direction | Servo Range | World Coverage |
|--------|----------|---------|----------------|-------------|----------------|
| 1 | (-1.5, -1.5, 0.5) | Southwest | 45° (toward center) | -90° to +90° | -45° to 135° |
| 2 | (1.5, -1.5, 0.5) | Southeast | 135° (toward center) | -90° to +90° | 45° to 225° |
| 3 | (1.5, 1.5, 0.5) | Northeast | 225° (toward center) | -90° to +90° | 135° to 315° |
| 4 | (-1.5, 1.5, 0.5) | Northwest | 315° (toward center) | -90° to +90° | 225° to 45° |

### Critical Positioning Requirements
1. **Height**: All sensors at exactly 0.5m above ground
2. **Accuracy**: ±2cm positioning tolerance
3. **Level**: Sensors must be perfectly horizontal
4. **Stability**: Mounts must prevent vibration during servo movement
5. **Clear Field**: No obstructions in sensor coverage areas

### Servo Home Directions
- **Home Direction**: Calculated automatically as angle from sensor position toward field center (0,0)
  - Sensor 1 (SW): 45° (points toward field center)
  - Sensor 2 (SE): 135° (points toward field center)
  - Sensor 3 (NE): 225° (points toward field center)
  - Sensor 4 (NW): 315° (points toward field center)
- **Servo Range**: ±90° from home direction (servo angles: -90° to +90°)
- **Movement Pattern**: Continuous back-and-forth sweep
- **Speed**: 180°/second (configurable in `config.yaml`)
- **Servo Position 0°**: Points directly toward field center
- **Coverage**: Each sensor covers 180° arc centered on field

## Communication Protocol

### Serial Settings
```yaml
communication:
  serial:
    baudrate: 115200
    timeout: 0.1  # seconds
  message_rate: 80  # milliseconds between messages per sensor
```

### Connection Details
- **Port**: COM3 (Windows) or /dev/ttyUSB0 (Linux) - adjust as needed
- **Baud Rate**: 115200
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Flow Control**: None

## Message Format

### JSON Message Structure
Each sensor sends JSON messages via serial when a detection occurs:

```json
{
    "sensor_id": 1,
    "angle": -27.5,
    "distance": 1.23,
    "local_position": [0, 0],
    "timestamp": 1634567890.123
}
```

### Field Descriptions
| Field | Type | Description | Units | Range |
|-------|------|-------------|-------|-------|
| `sensor_id` | int | Unique sensor identifier | - | 1-4 |
| `angle` | float | **Servo angle relative to home direction** | degrees | -90 to +90 |
| `distance` | float | Distance to target | meters | 0.02 to 2.5 |
| `local_position` | array | Unused (legacy) | - | [0, 0] |
| `timestamp` | float | Unix timestamp | seconds | Current time |

### Critical Message Requirements

#### Servo Angle Calculation
The `angle` field must contain the **servo position relative to home direction**:

```python
# Calculate servo angle from current servo position and home direction
servo_angle = current_servo_position - home_direction

# Example: Sensor 1 (home=45°) with servo pointing at 18° world angle
servo_angle = 18.0 - 45.0 = -27.0 degrees

# Servo angle range is always -90° to +90° from home direction
servo_angle = max(-90, min(90, servo_angle))
```

#### Hardware Implementation
```python
# In your Arduino code:
const float HOME_DIRECTION = 45.0;  // For sensor 1 (calculated from position)
float current_servo_world_angle = HOME_DIRECTION + servo_position;

# Send servo position directly (NOT world angle):
float servo_angle_to_send = servo_position;  // This is what goes in JSON message
```

#### Coordinate Transformation Pipeline
The system automatically converts servo angles to world coordinates:
1. **Hardware sends**: Servo angle (-90° to +90°)
2. **Preprocessor calculates**: `world_angle = home_direction + servo_angle`
3. **Preprocessor converts**: `(x_world, y_world) = measurement_to_world(distance, world_angle, sensor_config)`

## Configuration Files

### config.yaml Structure
```yaml
# Field dimensions (meters)
field:
  width: 3.0
  height: 3.0
  center:
    x: 0.0
    y: 0.0

# Sensor configuration
sensors:
  - id: 1
    name: "Sensor_1"
    position:
      x: -1.5  # Southwest corner
      y: -1.5
      z: 0.5   # Height above ground
    servo:
      pan_range: [-90, 90]     # Degrees relative to home direction
      pan_speed: 180.0         # Degrees per second
    max_range: 2.5  # Maximum detection range in meters

  - id: 2
    name: "Sensor_2"
    position:
      x: 1.5   # Southeast corner
      y: -1.5
      z: 0.5
    servo:
      pan_range: [-90, 90]
      pan_speed: 180.0
    max_range: 2.5

  - id: 3
    name: "Sensor_3"
    position:
      x: 1.5   # Northeast corner
      y: 1.5
      z: 0.5
    servo:
      pan_range: [-90, 90]
      pan_speed: 180.0
    max_range: 2.5

  - id: 4
    name: "Sensor_4"
    position:
      x: -1.5  # Northwest corner
      y: 1.5
      z: 0.5
    servo:
      pan_range: [-90, 90]
      pan_speed: 180.0
    max_range: 2.5

# Tracking parameters
tracking:
  kalman:
    process_noise: 0.1
    measurement_noise_distance: 0.02  # 2cm standard deviation
    measurement_noise_pan: 2.0        # 2 degrees standard deviation
    initial_covariance: 1.0
    
  association:
    max_distance: 0.5  # meters
    min_confidence: 0.3
    
  track_management:
    init_threshold: 2      # measurements needed to create track
    delete_threshold: 5    # missed updates before deletion
    max_tracks: 10

# Communication settings
communication:
  serial:
    baudrate: 115200
    timeout: 0.1
  message_rate: 80  # milliseconds between messages per sensor
```

## Hardware Configuration

### Microcontroller Code Template (Arduino)
```cpp
#include <Servo.h>
#include <NewPing.h>

// Pin definitions
#define TRIGGER_PIN 12
#define ECHO_PIN 11
#define SERVO_PIN 9

// Sensor configuration (MUST match config.yaml)
const int SENSOR_ID = 1;  // Change for each sensor (1-4)
const float SENSOR_X = -1.5;  // Sensor position X
const float SENSOR_Y = -1.5;  // Sensor position Y
// NOTE: HOME_DIRECTION not needed - system calculates automatically
// Sensor 1: 45°, Sensor 2: 135°, Sensor 3: 225°, Sensor 4: 315°

// Hardware objects
Servo panServo;
NewPing sonar(TRIGGER_PIN, ECHO_PIN, 250);  // Max distance 2.5m

// Servo control
float currentAngle = 0.0;  // Current servo angle relative to home
float servoSpeed = 180.0;  // degrees per second
int servoDirection = 1;    // 1 for forward, -1 for reverse
unsigned long lastServoUpdate = 0;
const int SERVO_UPDATE_MS = 50;  // Update every 50ms

void setup() {
  Serial.begin(115200);
  panServo.attach(SERVO_PIN);
  
  // Move to home position
  panServo.write(90);  // Assuming 90 is home position
  delay(1000);
}

void loop() {
  // Update servo position
  updateServo();
  
  // Take measurement
  float distance = sonar.ping_cm() / 100.0;  // Convert to meters
  
  if (distance > 0.02 && distance < 2.5) {  // Valid measurement
    // Send servo angle directly (NOT world angle)
    // The preprocessor will handle conversion to world coordinates
    sendMeasurement(distance, currentAngle);  // currentAngle is servo position (-90 to +90)
  }
  
  delay(50);  // Small delay between measurements
}

void updateServo() {
  unsigned long now = millis();
  if (now - lastServoUpdate >= SERVO_UPDATE_MS) {
    // Calculate new angle
    float angleIncrement = (servoSpeed * SERVO_UPDATE_MS / 1000.0) * servoDirection;
    currentAngle += angleIncrement;
    
    // Reverse direction at limits
    if (currentAngle >= 90.0) {
      currentAngle = 90.0;
      servoDirection = -1;
    } else if (currentAngle <= -90.0) {
      currentAngle = -90.0;
      servoDirection = 1;
    }
    
    // Convert to servo position (assuming 90 is center)
    int servoPos = 90 + (int)currentAngle;
    panServo.write(servoPos);
    
    lastServoUpdate = now;
  }
}

void sendMeasurement(float distance, float servoAngle) {
  float timestamp = millis() / 1000.0;  // Convert to seconds
  
  Serial.print("{\"sensor_id\":");
  Serial.print(SENSOR_ID);
  Serial.print(",\"angle\":");
  Serial.print(servoAngle, 2);  // Send servo angle directly (-90 to +90)
  Serial.print(",\"distance\":");
  Serial.print(distance, 3);
  Serial.print(",\"local_position\":[0,0]");
  Serial.print(",\"timestamp\":");
  Serial.print(timestamp, 3);
  Serial.println("}");
}
```

### Key Hardware Configuration Points
1. **Servo Calibration**: Ensure 90° servo position (0° relative angle) points toward field center
2. **Servo Range**: Servo sweeps from -90° to +90° relative to home direction
3. **Message Format**: Send servo angles (-90° to +90°), NOT world coordinate angles
4. **Timing**: Maintain consistent sweep speed and measurement timing
5. **Noise Handling**: Filter invalid measurements (too close/far, noise spikes)
6. **Coordinate Conversion**: Let the preprocessor handle servo-to-world coordinate transformation

## Testing and Validation

### 1. Communication Test
```bash
# Test serial communication
python scripts/main.py --debug

# Expected output: JSON messages from all sensors
```

### 2. Position Validation
Place a known object at field center (0, 0) and verify all sensors can detect it at servo angle 0°.

Expected servo angles for object at field center (0, 0):
- Sensor 1: 0.0° (servo pointing toward center)
- Sensor 2: 0.0° (servo pointing toward center)
- Sensor 3: 0.0° (servo pointing toward center)  
- Sensor 4: 0.0° (servo pointing toward center)

The preprocessor will convert these to the correct world coordinates automatically.

### 3. Coverage Test
Use the simulator to validate sensor coverage:
```bash
python scripts/main.py --sim --viz --debug
```

### 4. Coordinate Transform Validation
```python
from src.astraPro.preprocessor.transforms import measurement_to_world
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test known measurement
sensor_1 = config['sensors'][0]
field = config['field']

# Object at field center should give (0, 0)
x, y = measurement_to_world(2.12, 45.0, sensor_1, field)
print(f"Object at center: ({x:.2f}, {y:.2f})")  # Should be ~(0.0, 0.0)
```

## Troubleshooting

### Common Issues

#### 1. Incorrect Angles
**Symptoms**: Objects appear at wrong positions
**Causes**: 
- Servo home direction miscalibrated
- Angle calculation error in firmware
- Sensor mounted at wrong orientation

**Solutions**:
- Verify servo points toward field center at home position
- Check angle calculation: `world_angle = home_direction + servo_angle`
- Use protractor to verify physical sensor orientation

#### 2. Poor Detection Range
**Symptoms**: Sensors don't detect objects in expected range
**Causes**:
- Ultrasonic sensor malfunction
- Electrical interference
- Object material doesn't reflect ultrasound well

**Solutions**:
- Test sensor with Arduino serial monitor
- Check power supply voltage and stability
- Use reflective target materials

#### 3. Communication Issues
**Symptoms**: No data received from sensors
**Causes**:
- Wrong COM port
- Baud rate mismatch
- Faulty USB connection

**Solutions**:
- Verify correct COM port in Device Manager
- Confirm baud rate matches (115200)
- Try different USB cable/port

#### 4. Fusion Problems
**Symptoms**: Multiple targets for single object
**Causes**:
- Sensors not properly synchronized
- Timing issues between measurements
- Position calibration errors

**Solutions**:
- Adjust `max_distance` in triangulation engine
- Verify sensor positions match config.yaml exactly
- Check measurement timestamps are realistic

### Calibration Procedures

#### 1. Physical Positioning
1. Measure field dimensions precisely
2. Mark corner positions with tape
3. Use laser level for height consistency
4. Verify perpendicular sensor mounting

#### 2. Servo Calibration
1. Connect each sensor individually
2. Set servo to 90° position
3. Verify sensor points toward field center
4. Adjust mounting if necessary

#### 3. Angle Verification
1. Place object at known positions
2. Compare detected angles with calculated values
3. Adjust home_direction values if needed
4. Test all quadrants of the field

### Performance Optimization

#### 1. Measurement Rate
- Default: 80ms between measurements per sensor
- Faster rates: Better tracking but more processing
- Slower rates: Less processing but delayed tracking

#### 2. Servo Speed
- Default: 180°/second
- Faster speeds: Better coverage but less measurement time per angle
- Slower speeds: More measurements per angle but slower coverage

#### 3. Filtering
- Distance: Filter measurements outside 0.02m - 2.5m range
- Angle: Smooth servo movement to reduce jitter
- Time: Remove measurements older than 500ms

## Summary

This guide provides all necessary information to set up your physical sensor system to match the AstraPro simulator. Key points:

1. **Exact positioning**: Sensors must be positioned precisely at corner coordinates
2. **Proper communication**: JSON messages with correct world coordinate angles
3. **Servo calibration**: Home directions must point toward field center
4. **Configuration matching**: config.yaml must reflect actual hardware setup
5. **Systematic testing**: Validate each component before full system integration

For additional support, refer to the source code in `src/astraPro/` and use the simulation mode for testing and validation.