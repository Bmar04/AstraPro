# Tracking System Implementation Plan

## Overview
This document outlines the implementation plan for adding a tracking system with Kalman filtering to convert fused target detections into persistent tracks with state estimation.

## Current State
- ✅ Multi-sensor fusion working (fused target detections)
- ✅ Realistic multi-target simulation (3 moving objects)
- ✅ Coordinate transforms and preprocessing pipeline
- ❌ No temporal continuity between detections
- ❌ No state estimation (position, velocity, acceleration)
- ❌ No track management (creation, deletion, association)

## Goal
Transform instantaneous target detections into persistent tracks with:
- **State Estimation**: Position, velocity, and uncertainty estimates
- **Prediction**: Where targets will be in the next time step
- **Data Association**: Match new detections to existing tracks
- **Track Management**: Create new tracks, delete lost tracks

## Architecture Overview

```
Raw Measurements → Preprocessing → Triangulation → Tracking → Tracks Output
                                    (Fusion)      (Kalman)
                                      ↓              ↓
                              Fused Targets → Track Manager → Persistent Tracks
                                              ↓
                                        Data Association
                                              ↓
                                        Kalman Filters
```

## Implementation Plan

### Phase 1: Kalman Filter Foundation (2-3 hours)
**Goal**: Implement basic Kalman filter for single target

#### Tasks:
1. **Create Kalman Filter Class** (`tracker/kalman.py`)
   - State vector: `[x, y, vx, vy]` (position + velocity)
   - Constant velocity motion model
   - Process noise and measurement noise parameters
   - Predict and update methods

2. **Track State Model** (`tracker/models.py`)
   - Track class with ID, state, covariance, timestamp
   - Track status (new, confirmed, lost)
   - Confidence and age tracking

3. **Basic Test**
   - Unit test with synthetic measurements
   - Verify prediction and update cycles

**Deliverables**:
- Working Kalman filter that can track a single target
- State estimation with uncertainty propagation

### Phase 2: Data Association (2-3 hours)
**Goal**: Match fused targets to existing tracks

#### Tasks:
1. **Association Algorithms** (`tracker/association.py`)
   - Nearest neighbor association
   - Mahalanobis distance gating
   - Hungarian algorithm for optimal assignment (optional)

2. **Gating and Validation**
   - Distance thresholds for association
   - Innovation covariance calculation
   - Reject unlikely associations

3. **Association Test**
   - Test with 2 crossing targets
   - Verify correct assignment during crossings

**Deliverables**:
- Robust association that maintains track identity
- Handles crossing and nearby targets

### Phase 3: Track Management (2-3 hours)
**Goal**: Create, maintain, and delete tracks automatically

#### Tasks:
1. **Track Manager** (`tracker/track_manager.py`)
   - Track creation from unassociated detections
   - Track confirmation (N out of M hits)
   - Track deletion (missed detections)
   - Track ID assignment

2. **Track Lifecycle**
   - States: Tentative → Confirmed → Lost → Deleted
   - Configurable thresholds for state transitions
   - Track quality scoring

3. **Multi-Target Integration**
   - Handle variable number of targets
   - Simultaneous track creation/deletion
   - Memory management

**Deliverables**:
- Automatic track management
- Stable tracking of multiple targets

### Phase 4: Integration and Tuning (1-2 hours)
**Goal**: Integrate tracking into main pipeline and optimize

#### Tasks:
1. **Pipeline Integration**
   - Modify `main.py` to use tracking system
   - Connect fused targets to track manager
   - Output format for tracks vs detections

2. **Parameter Tuning**
   - Process noise (how much targets can accelerate)
   - Measurement noise (sensor accuracy)
   - Association gates and thresholds
   - Track management parameters

3. **Performance Testing**
   - Test with simulation scenarios
   - Measure tracking accuracy
   - Stress test with many targets

**Deliverables**:
- Complete tracking pipeline
- Tuned parameters for your application
- Performance metrics

## Technical Specifications

### Kalman Filter State Model
```
State Vector: x = [px, py, vx, vy]
- px, py: Position (meters)
- vx, vy: Velocity (m/s)

Motion Model (Constant Velocity):
x(k+1) = F * x(k) + w(k)

F = [1  0  dt  0 ]  where dt = time step
    [0  1   0 dt ]
    [0  0   1  0 ]
    [0  0   0  1 ]

Process Noise Q: Accounts for acceleration uncertainty
Measurement Model: z = [px, py] (only observe position)
```

### Key Parameters to Tune
- **Process Noise**: How much can targets accelerate? (σ_a = 1-5 m/s²)
- **Measurement Noise**: Sensor accuracy (σ_r = 0.1-0.5 m)
- **Association Gate**: Maximum distance for association (3-5σ)
- **Track Confirmation**: N=3 detections out of M=5 attempts
- **Track Deletion**: Miss target for T=5-10 time steps

### Data Structures
```python
@dataclass
class TrackState:
    id: int
    state: np.array  # [px, py, vx, vy]
    covariance: np.array  # 4x4 uncertainty matrix
    timestamp: float
    status: TrackStatus  # NEW, CONFIRMED, LOST
    hits: int
    misses: int
    confidence: float

@dataclass 
class Track:
    id: int
    state_history: List[TrackState]
    created_time: float
    last_update: float
    
    def predict(self, dt: float) -> TrackState
    def update(self, detection: FusedTarget) -> TrackState
```

## File Structure
```
tracker/
├── __init__.py
├── kalman.py           # Kalman filter implementation
├── models.py           # Track data structures  
├── association.py      # Data association algorithms
├── track_manager.py    # Track lifecycle management
└── tracker.py          # Main tracking interface

tests/
├── test_kalman.py      # Kalman filter unit tests
├── test_association.py # Association tests
└── test_tracking.py    # Integration tests
```

## Success Metrics
1. **Track Continuity**: Same track ID maintained for >90% of target lifetime
2. **Position Accuracy**: RMS error <20cm compared to ground truth
3. **Velocity Estimation**: Velocity estimates within 20% of true velocity
4. **Association Accuracy**: <5% wrong associations during crossing scenarios
5. **Real-time Performance**: Process 20 Hz updates with <10ms latency

## Testing Strategy
1. **Unit Tests**: Individual components (Kalman, association)
2. **Synthetic Data**: Known ground truth for accuracy measurement
3. **Simulation Tests**: Use existing 3-target simulation
4. **Stress Tests**: Many targets, fast motion, sensor dropouts
5. **Edge Cases**: Target appearance/disappearance, occlusion

## Risk Mitigation
- **Track Loss**: Implement track prediction during missed detections
- **False Tracks**: Require confirmation before outputting tracks
- **Computational Load**: Optimize matrix operations, limit max tracks
- **Parameter Sensitivity**: Provide good defaults, auto-tuning options

## Next Steps
1. Start with Phase 1: Implement basic Kalman filter
2. Test with single target from simulation
3. Gradually add complexity (association, management)
4. Integrate with existing fusion pipeline
5. Tune and optimize for your specific use case

This plan will give you a production-ready tracking system that converts your excellent fused detections into persistent, state-estimated tracks.