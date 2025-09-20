"""
Simple triangulation engine.
"""

import time
import math
from .models import FusedTarget, FusionResult, FusionConfig, create_target


class TriangulationEngine:
    
    def __init__(self, max_distance=0.5, min_confidence=0.3):
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.config = FusionConfig(max_distance, min_confidence)
        self.measurements = []
        self.last_fusion = time.time()
    
    def add_measurement(self, measurement):
        # Remove old measurements
        cutoff = time.time() - 0.5  # 500ms timeout
        self.measurements = [m for m in self.measurements if m.timestamp > cutoff]
        self.measurements.append(measurement)
    
    def should_fuse(self):
        return (time.time() - self.last_fusion) > 0.05  # 50ms - faster for rotating sensors
    
    def fuse_targets(self):
        if not self.measurements:
            return FusionResult([])
        
        # Group nearby measurements
        groups = self._group_measurements()
        
        # Create targets
        targets = []
        for group in groups:
            target = create_target(group, self.config)
            if target.confidence >= self.min_confidence:
                targets.append(target)
        
        self.measurements.clear()
        self.last_fusion = time.time()
        return FusionResult(targets)
    
    def _group_measurements(self):
        groups = []
        used = set()
        
        for i, m1 in enumerate(self.measurements):
            if i in used:
                continue
            
            group = [m1]
            used.add(i)
            
            for j, m2 in enumerate(self.measurements):
                if j in used:
                    continue
                
                dist = math.sqrt((m1.x_world - m2.x_world)**2 + (m1.y_world - m2.y_world)**2)
                if dist <= self.max_distance:
                    group.append(m2)
                    used.add(j)
            
            groups.append(group)
        
        return groups