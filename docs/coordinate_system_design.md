# Coordinate System Design & Implementation Plan

## System Overview

The sonar fusion system uses servo-controlled ultrasonic sensors at each corner of a 6x6m field. Each sensor measures in polar coordinates (range, pan_angle) and we must transform these measurements into a unified world coordinate system centered at the field's center.

## Coordinate Systems

### 1. World Coordinate System
- **Origin**: Center of the 6x6m field (3,3) relative to corner
- **X-axis**: Pointing East (positive to the right when looking North)
- **Y-axis**: Pointing North (positive upward on field diagram)
- **Z-axis**: Pointing up (height above ground)
- **Units**: Meters

```
Field Layout (Top View):
                 Y (North)
                    ↑
    Sensor 4        |        Sensor 3
    (-3,3)          |          (3,3)
         ╭──────────┼──────────╮
         │          │          │
         │          │          │
         │          │          │
    ──────┼──────────O──────────┼────── X (East)
         │        (0,0)        │
         │                     │
         │                     │
         ╰─────────────────────╯
    Sensor 1                 Sensor 2
    (-3,-3)                   (3,-3)
```

### 2. Sensor Local Coordinate System
- **Origin**: At each sensor position
- **X-axis**: Points toward field center when servo at home position (pan=0°)
- **Y-axis**: Perpendicular to X-axis (90° CCW from X)
- **Z-axis**: Points up
- **Pan angle**: Measured from sensor's local X-axis (CCW positive)

```
Sensor Local Coordinate System (Example: Sensor 1 at (-3,-3)):

    Sensor 1 View:
         Y_local ↑
                 │ 
                 │    Target
                 │   ● (r, θ)
                 │  ╱
                 │ ╱ 
                 │╱θ (pan angle)
    ─────────────●─────────────→ X_local
               Sensor            (toward field center)
                                 
    Pan Angle Convention:
    • 0°: Pointing toward field center
    • +90°: CCW from center direction  
    • -90°: CW from center direction
```

### 3. Measurement Space (Polar)
Each sensor provides:
- **r**: Range to target (meters)
- **θ**: Pan angle of servo (degrees, relative to sensor's home position)
- **No tilt**: Sensors are horizontally mounted

## Transformation Chain

```
Polar Measurement → Sensor Local Cartesian → World Cartesian
    (r, θ)       →      (x_local, y_local)  →    (x_world, y_world)
```

## Mathematical Derivations

### Step 1: Polar to Sensor Local Cartesian

For a measurement (r, θ) from sensor i:

```
x_local = r × cos(θ)
y_local = r × sin(θ)
```

Where θ is the pan angle in radians.

### Step 2: Sensor Local to World Coordinates

**Automatic Home Direction Calculation:**

Each sensor's home direction is automatically calculated from its position relative to the field center:

```python
def calculate_home_direction(sensor_pos, field_center):
    dx = field_center.x - sensor_pos.x
    dy = field_center.y - sensor_pos.y
    angle = atan2(dy, dx) * 180 / π
    return angle if angle >= 0 else angle + 360
```

**Examples (automatically calculated):**
- **Sensor 1** (-3,-3): dx=3, dy=3 → α₁ = 45°
- **Sensor 2** (+3,-3): dx=-3, dy=3 → α₂ = 135°  
- **Sensor 3** (+3,+3): dx=-3, dy=-3 → α₃ = 225°
- **Sensor 4** (-3,+3): dx=3, dy=-3 → α₄ = 315°

**General transformation matrix:**
```
[x_world]   [cos(α)  -sin(α)] [x_local]   [px]
[y_world] = [sin(α)   cos(α)] [y_local] + [py]

Where α is calculated automatically from sensor and field positions
```

### Complete Transformation (Automatic)

For any sensor with measurement (r, θ), the transformation is completely automatic:

```python
from preprocessor.transforms import measurement_to_world

# All calculations happen automatically based on config.yaml positions
x_world, y_world = measurement_to_world(
    range_m=r, 
    pan_angle_deg=θ,
    sensor_config=sensor_config,
    field_config=field_config
)
```

**Internal process:**
```python
# Step 1: Polar to local cartesian
x_local = r * cos(θ)
y_local = r * sin(θ)

# Step 2: Calculate home direction automatically
α = calculate_home_direction(sensor_pos, field_center)

# Step 3: Local to world transformation
x_world = cos(α) * x_local - sin(α) * y_local + px
y_world = sin(α) * x_local + cos(α) * y_local + py
```

## Implementation Plan - Day 1

### Morning (2 hours): Core Transform Implementation

1. **Update config.yaml** (15 min)
   - Change sensor positions to world coordinates with center origin
   - Add sensor home direction angles

2. **Implement `preprocessor/transforms.py`** (90 min) - ✅ COMPLETED
   - `calculate_home_direction()` - automatic direction calculation
   - `polar_to_cartesian(range, pan_angle)` 
   - `sensor_local_to_world()` with automatic direction
   - `measurement_to_world()` - complete automatic transformation
   - Inverse functions for testing and validation

3. **Create test cases** (15 min)
   - Known positions for validation
   - Edge cases (sensor pointing directions)

### Afternoon (3 hours): Comprehensive Testing

1. **Unit Tests** (60 min)
   - Test each transformation function individually
   - Verify inverse transforms work correctly
   - Test with known analytical solutions

2. **Integration Tests** (60 min)
   - End-to-end: polar measurement → world coordinates
   - Test all four sensors with same target position
   - Verify geometric consistency between sensors

3. **Visualization Setup** (60 min)
   - Create simple plot showing sensor positions
   - Plot transformation results for verification
   - Add coordinate system visualization

### Evening (1 hour): Validation & Documentation

1. **Cross-validation** (30 min)
   - Place virtual target at known positions
   - Verify all sensors report consistent world coordinates
   - Check boundary conditions

2. **Documentation** (30 min)
   - Document coordinate system conventions
   - Add usage examples
   - Create troubleshooting guide

## Test Cases for Validation

### Known Position Tests
1. **Field Center (0,0)**: All sensors should report same world coordinates
2. **Field Corners**: Test near sensor positions
3. **Field Edges**: Test boundary conditions

### Geometric Consistency Tests
1. **Same Target, Multiple Sensors**: World coordinates should match within noise
2. **Sensor Cross-check**: Use triangulation to verify single measurements
3. **Movement Test**: Track target across field, ensure smooth coordinate transitions

### Edge Cases
1. **Maximum Range**: Test at sensor range limits
2. **Pan Limits**: Test at ±90° pan angles  
3. **Sensor Occlusion**: When target outside sensor's view cone

## Expected Outputs

After Day 1 implementation:
- ✅ Accurate coordinate transforms for all sensor positions
- ✅ Validated with analytical test cases  
- ✅ Visual verification of coordinate system
- ✅ Ready for Day 2 uncertainty propagation
- ✅ <1ms transformation time per measurement

## Error Analysis

### Sources of Error
1. **Measurement Noise**: ±2cm distance, ±2° pan angle
2. **Servo Positioning**: Mechanical backlash, encoder resolution
3. **Sensor Mounting**: Position/orientation uncertainty
4. **Timing**: Servo position vs measurement timestamp

### Mitigation Strategies
1. **Calibration**: Precise sensor position measurement
2. **Filtering**: Remove obvious outliers before transformation
3. **Uncertainty Propagation**: Track covariance through transforms
4. **Validation**: Cross-sensor consistency checks

This coordinate system design ensures mathematical rigor while being practical for implementation and testing.