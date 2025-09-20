# AstraPro Multi-Target Tracking System

Professional real-time multi-sensor tracking system with advanced sensor fusion and Kalman filtering for persistent target tracking in a 3x3 meter field.

## Quick Start

### Basic Simulation
```bash
python scripts/main.py --sim --debug
```

### AI Tracking with Visualization
```bash
python scripts/main.py --sim --aitracker --debug --viz
```

### Full Featured Session
```bash
python scripts/main.py --sim --aitracker --viz --log --debug
```

### Analysis Tool
```bash
# Analyze tracking session
python scripts/analyze_tracking.py data/ tracking_session_20250119_143052

# Custom output directory
python scripts/analyze_tracking.py data/ session_name --output my_analysis/

# Compare multiple sessions
python scripts/analyze_tracking.py data/ session1 --compare session2 session3
```

## Main Application Flags

| Flag | Description |
|------|-------------|
| `--sim` | Enable simulation mode (generates synthetic sensor data) |
| `--debug` | Enable debug output and detailed logging |
| `--aitracker` | Use AI-based tracking with Kalman filters |
| `--tracker` | Use manual tracking system (implementation required) |
| `--viz` | Enable real-time visualization window |
| `--log` | Enable CSV data logging to data/ directory |

### Flag Combinations

```bash
# Simulation with debug output
python scripts/main.py --sim --debug

# AI tracking with all features
python scripts/main.py --sim --aitracker --viz --log --debug

# Hardware mode with AI tracking
python scripts/main.py --aitracker --debug

# Simulation only (minimal)
python scripts/main.py --sim
```

## Analysis Tool Flags

| Flag/Argument | Required | Description |
|---------------|----------|-------------|
| `data_dir` | Yes | Directory containing CSV files |
| `session_name` | Yes | Session name prefix of CSV files |
| `--output`, `-o` | No | Output directory (default: analysis) |
| `--compare` | No | Compare with other sessions (space-separated) |

### Analysis Examples

```bash
# Basic analysis
python scripts/analyze_tracking.py data/ tracking_session_20250119_143052

# Custom output location
python scripts/analyze_tracking.py data/ session_name -o results/

# Session comparison
python scripts/analyze_tracking.py data/ session1 --compare session2 session3

# Cross-directory comparison
python scripts/analyze_tracking.py data/ session1 --compare other_data/:session2
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individual packages
pip install numpy pandas matplotlib seaborn PyYAML plotly
```

## System Overview

The system processes data through these stages:

1. **Raw Measurements** - Distance/angle data from 4 rotating corner sensors
2. **Preprocessing** - Coordinate transformation and noise filtering
3. **Triangulation** - Multi-sensor fusion for target detection
4. **Tracking** - Kalman filtering with data association
5. **Visualization** - Real-time display and analysis tools

### Key Features

- Multi-sensor triangulation fusion for 3x3 meter field
- Rotating sensor simulation with realistic timing
- Kalman filter tracking with state estimation
- Real-time visualization and comprehensive logging
- Interactive web-based trajectory analysis
- Performance scorecard generation with ground truth

## Simulation Modes

### Simulation Mode (`--sim`)
- Generates synthetic sensor measurements
- 4 rotating sensors at field corners (±1.5m, ±1.5m)
- Configurable movement patterns (diagonal, circular, random)
- Realistic sensor noise and timing constraints

### Hardware Mode (default)
- Reads from serial interface (COM3)
- Processes real sensor measurements
- Requires physical sensor hardware

## Tracking Systems

### AI Tracker (`--aitracker`)
- Kalman filter with state vector [x, y, vx, vy]
- Nearest neighbor data association with gating
- Track lifecycle: NEW → CONFIRMED → LOST
- Configurable parameters for different scenarios

### Manual Tracker (`--tracker`)
- User-implemented tracking algorithms
- Implement TrackerSystem class in tracker/ directory
- Uncomment import lines in main.py

## Visualization (`--viz`)

Real-time display shows:
- Field layout with sensor positions
- Target detections (light blue circles)
- Active tracks with colored trails
- Velocity vectors and status indicators
- Track statistics and counts

## Data Logging (`--log`)

Creates timestamped CSV files in data/ directory:
- `{session}_tracks.csv` - Track states over time
- `{session}_detections.csv` - Fused target detections
- `{session}_stats.csv` - System performance metrics
- `{session}_ground_truth.csv` - Simulation ground truth

## Analysis Output

The analysis tool generates:
- Trajectory plots and speed analysis
- Performance metrics and scorecards
- Interactive HTML visualization
- Comprehensive text reports

## Configuration

### Performance Tuning

Reduce false tracks by adjusting parameters in `scripts/main.py`:

```python
# Stricter fusion filtering
fusion_engine = TriangulationEngine(max_distance=0.3, min_confidence=0.5)

# Tighter association gate
tracking_system = TrackerSystem(gate_threshold=2.0)
```

### Movement Patterns

Edit `src/astraPro/io_module/simulator.py` line 9:

```python
DEFAULT_SCENARIO = "all"  # Options: all, diagonal, circle, random
```

## File Structure

```
astraPro/
├── scripts/
│   ├── main.py                 # Main application entry
│   └── analyze_tracking.py     # Analysis tool
├── src/astraPro/
│   ├── io_module/             # Simulation and data input
│   ├── preprocessor/          # Sensor fusion and processing
│   ├── aitracker/            # Kalman filter tracking
│   └── visualizer/           # Display and logging
├── config.yaml               # System configuration
└── data/                    # CSV output directory
```

## Technical Specifications

- **Field Size**: 3x3 meters
- **Sensors**: 4 rotating sensors, 2.5m range, 20° FOV
- **Update Rate**: 20 Hz simulation, 50ms fusion timing
- **Tracking**: Gate threshold 3.5, max fusion distance 1.0m
- **Noise Model**: 1cm distance, 1° angle uncertainty

## Troubleshooting

### Common Issues
- **Import Errors**: Install all dependencies from requirements.txt
- **Serial Connection**: Check COM port for hardware mode
- **No Confirmed Tracks**: Adjust tracking parameters
- **Performance**: Disable --viz for maximum speed

### Debug Commands
```bash
# System investigation
python complete_debug.py

# Pipeline debugging
python debug_pipeline.py

# Coordinate testing
python test_coordinates.py
```

## API Reference

### Core Components

```python
# Tracking system
from astraPro.aitracker.tracker import TrackerSystem
tracker = TrackerSystem(gate_threshold=3.5)

# Sensor fusion
from astraPro.preprocessor.triangulation import TriangulationEngine
fusion = TriangulationEngine(max_distance=1.0, min_confidence=0.05)

# Visualization
from astraPro.visualizer.working_viz import WorkingVisualizer
viz = WorkingVisualizer()

# Data logging
from astraPro.visualizer.data_logger import TrackDataLogger
logger = TrackDataLogger()
```

For detailed implementation examples and advanced configuration, refer to the source code and configuration files.