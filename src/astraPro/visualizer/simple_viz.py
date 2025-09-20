"""
Dead simple real-time visualizer that actually works.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

class SimpleVisualizer:
    def __init__(self):
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Live Tracking')
        
        # Draw field boundary
        self.ax.plot([-1.5, 1.5, 1.5, -1.5, -1.5], [-1.5, -1.5, 1.5, 1.5, -1.5], 'k-', linewidth=2)
        
        # Draw sensors
        sensors = [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5), (0.0, 1.5)]
        for i, (x, y) in enumerate(sensors):
            self.ax.plot(x, y, 'ys', markersize=10)
            self.ax.text(x+0.1, y+0.1, f'S{i+1}', fontsize=10)
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def update(self, tracking_result, detections):
        """Update with new tracking data."""
        # Clear old tracks and detections
        for collection in self.ax.collections:
            collection.remove()
        for txt in self.ax.texts[4:]:  # Keep sensor labels, remove track labels
            txt.remove()
        
        # Plot detections as blue dots
        if detections:
            det_x = [d.x for d in detections]
            det_y = [d.y for d in detections]
            self.ax.scatter(det_x, det_y, c='lightblue', s=100, alpha=0.7, label='Detections')
        
        # Plot tracks
        if hasattr(tracking_result, 'tracks') and tracking_result.tracks:
            for track in tracking_result.tracks:
                pos = track.get_position()
                
                # Color based on status
                if hasattr(track, 'status'):
                    if str(track.status) == 'TrackStatus.CONFIRMED':
                        color = 'red'
                        size = 150
                    elif str(track.status) == 'TrackStatus.NEW':
                        color = 'orange'
                        size = 100
                    else:  # LOST
                        color = 'gray'
                        size = 80
                else:
                    color = 'red'
                    size = 150
                
                # Plot track
                self.ax.scatter(pos[0], pos[1], c=color, s=size, alpha=0.8)
                self.ax.text(pos[0]+0.05, pos[1]+0.05, f'{track.id}', fontsize=12, fontweight='bold')
        
        # Update title
        active_count = len(tracking_result.tracks) if hasattr(tracking_result, 'tracks') else 0
        confirmed_count = len([t for t in tracking_result.tracks if 'CONFIRMED' in str(getattr(t, 'status', ''))])
        detection_count = len(detections) if detections else 0
        
        self.ax.set_title(f'Live Tracking - Active: {active_count}, Confirmed: {confirmed_count}, Detections: {detection_count}')
        
        # Force update
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        plt.ioff()
        plt.close()