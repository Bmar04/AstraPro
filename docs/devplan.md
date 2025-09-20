# Sonar Fusion Development Timeline

## Development Philosophy
- **See Results Fast**: Get visualization working ASAP
- **Test Everything**: Verify each component before moving on
- **Incremental Complexity**: Start simple, add features gradually
- **Hardware Last**: Perfect the algorithms in simulation first

## Week 1: Foundation & Visualization
*Goal: See simulated balls moving on screen*

### Day 1-2: Project Setup & Basic Simulator
**Tasks:**
- [ ] Create project structure and git repo
- [ ] Write `config.yaml` with sensor positions
- [ ] Implement `io_module/data_models.py` (30 min)
- [ ] Create basic `io_module/simulator.py`:
  - [ ] Ball with constant velocity
  - [ ] Simple distance calculation (no angles yet)
  - [ ] Generate messages every 80ms per sensor

**Milestone:** Print simulated sensor readings to console

### Day 3-4: Basic Visualizer
**Tasks:**
- [ ] Implement `utils/visualizer.py`:
  - [ ] Draw 6x6m field
  - [ ] Show sensor positions
  - [ ] Plot raw distance measurements as circles
- [ ] Connect simulator to visualizer in simple `main.py`
- [ ] Add ball trajectory to visualization

**Milestone:** See animated ball moving with sensor circles

### Day 5: Add Angles & Noise
**Tasks:**
- [ ] Extend simulator:
  - [ ] Calculate pan/tilt angles
  - [ ] Add realistic noise (distance ±2cm, angle ±2°)
  - [ ] Implement ball bouncing off walls
- [ ] Update visualizer to show sensor "beams"

**Milestone:** Realistic-looking noisy sensor data

## Week 2: Coordinate Transforms & Basic Tracking
*Goal: Track single ball with Kalman filter*

### Day 6-7: Preprocessing Pipeline
**Tasks:**
- [ ] Implement `preprocessor/models.py` - Measurement class
- [ ] Write `preprocessor/transforms.py`:
  - [ ] `spherical_to_cartesian()` with proper trig
  - [ ] Simple uncertainty (fixed covariance for now)
- [ ] Test transforms with known values
- [ ] Update visualizer to show world coordinates

**Milestone:** See dots at actual ball position (with noise)

### Day 8-9: Simple Kalman Filter
**Tasks:**
- [ ] Implement `tracker/kalman.py`:
  - [ ] State: [x, y, vx, vy]
  - [ ] Constant velocity model
  - [ ] Basic predict/update cycle
- [ ] Create minimal `tracker/track_manager.py`:
  - [ ] Single track only
  - [ ] No association needed yet
- [ ] Visualize track vs ground truth

**Milestone:** Smooth track following noisy measurements

### Day 10: Tuning & Testing
**Tasks:**
- [ ] Tune Kalman filter parameters (process/measurement noise)
- [ ] Test with different trajectories:
  - [ ] Straight line
  - [ ] Circle
  - [ ] Sharp turns
- [ ] Add track uncertainty ellipse to visualization
- [ ] Measure tracking error (RMS)

**Milestone:** <10cm RMS error on simulated data

## Week 3: Multi-Sensor Fusion & Association
*Goal: Use all 4 sensors, track multiple balls*

### Day 11-12: Triangulation
**Tasks:**
- [ ] Implement `preprocessor/triangulation.py`:
  - [ ] Group simultaneous measurements
  - [ ] Average positions when multiple sensors see target
  - [ ] Weight by distance (closer = more accurate)
- [ ] Update visualization to show which sensors contribute
- [ ] Test improvement in accuracy

**Milestone:** Better accuracy when multiple sensors see target

### Day 13-14: Multi-Target Tracking
**Tasks:**
- [ ] Extend simulator for 2-3 balls
- [ ] Implement `tracker/association.py`:
  - [ ] Nearest neighbor association
  - [ ] Simple distance threshold
- [ ] Update track manager:
  - [ ] Multiple track support
  - [ ] Track creation for new targets
  - [ ] Track deletion for lost targets

**Milestone:** Track 2 balls simultaneously

### Day 15: Association Challenges
**Tasks:**
- [ ] Test crossing trajectories
- [ ] Add gating to reject bad associations
- [ ] Handle measurement-to-track conflicts
- [ ] Implement track confidence scoring

**Milestone:** Maintain correct tracks through crossings

## Week 4: Hardware Integration
*Goal: Track real golf ball*

### Day 16-17: Serial Interface
**Tasks:**
- [ ] Implement `io_module/serial_reader.py`:
  - [ ] Parse real message format
  - [ ] Handle connection errors
  - [ ] Message queue with timeout
- [ ] Create hardware test script
- [ ] Verify message format with actual Pico

**Milestone:** Receive real sensor data

### Day 18-19: Calibration & Testing
**Tasks:**
- [ ] Calibrate sensor positions precisely
- [ ] Measure actual sensor noise characteristics
- [ ] Update config with real parameters
- [ ] Test with stationary target at known positions
- [ ] Adjust coordinate transforms if needed

**Milestone:** Accurate position for stationary target

### Day 20: Live Tracking
**Tasks:**
- [ ] Track thrown/rolled ball
- [ ] Compare predicted vs actual landing position
- [ ] Tune filter parameters for real noise
- [ ] Test range limits and edge cases
- [ ] Record demo videos

**Milestone:** Track real moving ball in real-time

## Week 5: Polish & Documentation
*Goal: Professional deliverable*

### Day 21-22: Performance & Robustness
**Tasks:**
- [ ] Profile code, optimize bottlenecks
- [ ] Add comprehensive error handling
- [ ] Implement data logging for analysis
- [ ] Create startup/shutdown scripts
- [ ] Package management with setup.py

### Day 23-24: Documentation
**Tasks:**
- [ ] Write comprehensive README
- [ ] Document API with docstrings
- [ ] Create user guide with examples
- [ ] Theory document explaining algorithms
- [ ] Performance analysis report

### Day 25: Presentation Prep
**Tasks:**
- [ ] Create presentation slides
- [ ] Prepare live demos
- [ ] Record backup demo videos
- [ ] Practice explanation of algorithms
- [ ] Prepare for Q&A

## Critical Path Items

### Must Have (Core Functionality)
1. Simulator with realistic noise
2. Coordinate transforms
3. Single-target Kalman filter
4. Real-time visualization
5. Hardware serial interface

### Should Have (Professional Quality)
1. Multi-sensor fusion
2. Multi-target tracking
3. Proper association logic
4. Performance metrics
5. Comprehensive testing

### Nice to Have (If Time Permits)
1. Extended Kalman Filter for curved paths
2. 3D visualization
3. Web-based interface
4. Machine learning for motion prediction
5. Automatic calibration routine

## Daily Development Routine

### Morning (2 hours)
- Review previous day's work
- Implement new feature
- Write unit tests

### Afternoon (2 hours)  
- Test and debug
- Integrate with existing code
- Update visualization

### Evening (1 hour)
- Document progress
- Commit to git
- Plan next day

## Risk Mitigation

### If Behind Schedule:
- Skip multi-target (focus on perfect single-target)
- Use simple association (skip fancy algorithms)
- Minimize documentation (code should be self-documenting)
- Pre-record demos (avoid live demo failures)

### If Ahead of Schedule:
- Add EKF for better curved path tracking
- Implement multiple motion models
- Create GUI for parameter tuning
- Add more sophisticated visualization

## Success Metrics

### Week 1: ✓ See something moving
### Week 2: ✓ Tracking works in simulation  
### Week 3: ✓ Multi-sensor/target capability
### Week 4: ✓ Real hardware integration
### Week 5: ✓ Professional package

## Git Commit Strategy
- Daily commits minimum
- Feature branches for major components
- Tag working milestones (v0.1-simulator, v0.2-tracking, etc.)
- Never break main branch

This timeline gives you 5 weeks to build a professional tracking system with clear daily goals and flexibility to adjust based on progress.

file structure
 sonar-fusion/
├── README.md
├── requirements.txt
├── config.yaml                    # All configuration in one file
│
├── io_module/
│   ├── __init__.py
│   ├── serial_reader.py          # Hardware interface
│   ├── simulator.py              # Test data generator
│   └── data_models.py            # Message classes
│
├── preprocessor/
│   ├── __init__.py
│   ├── transforms.py             # Coordinate conversions
│   ├── triangulation.py          # Multi-sensor fusion
│   └── models.py                 # Measurement class
│
├── tracker/
│   ├── __init__.py
│   ├── kalman.py                 # Kalman filter
│   ├── association.py            # Data association
│   └── track_manager.py          # Main tracking logic
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py               # Math helpers
│   └── visualizer.py             # Real-time plotting
│
├── tests/
│   ├── test_transforms.py        # Critical tests only
│   └── test_tracking.py
│
└── main.py                       # Run everything