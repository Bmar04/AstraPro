"""
Working live tracker visualization - exactly like the original but fixed.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque

# Import TrackStatus properly
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from astraPro.aitracker.models import TrackStatus


class WorkingVisualizer:
    """
    Working real-time visualization of tracking data.
    """
    
    def __init__(self, field_size: Tuple[float, float] = (3.0, 3.0)):
        """Initialize working visualizer."""
        self.field_size = field_size
        
        # Track history storage
        self.track_histories: Dict[int, deque] = {}
        self.track_colors: Dict[int, str] = {}
        self.color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.color_index = 0
        
        # Current data
        self.current_tracks = []
        self.current_detections = []
        
        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()
        plt.show(block=False)
        plt.pause(0.1)
    
    def setup_plot(self):
        """Setup the matplotlib plot."""
        self.ax.clear()
        # Show just the 3x3m field boundaries (sensors at corners)
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Live Tracking Visualization')
        
        # Draw field boundary
        field_rect = patches.Rectangle(
            (-self.field_size[0]/2, -self.field_size[1]/2),
            self.field_size[0], self.field_size[1],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(field_rect)
        
        # Draw sensors at actual corners of 3x3m field with detection ranges
        sensor_positions = [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]
        for i, (x, y) in enumerate(sensor_positions):
            # Draw sensor detection circle
            circle = patches.Circle((x, y), 2.5, linewidth=1, edgecolor='gray', 
                                  facecolor='none', alpha=0.3, linestyle='--')
            self.ax.add_patch(circle)
            
            # Draw sensor
            self.ax.plot(x, y, 's', markersize=8, color='black', 
                        markerfacecolor='yellow', markeredgewidth=2)
            self.ax.annotate(f'S{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    def update(self, tracking_result, detections):
        """
        Update visualization with new tracking data.
        
        Args:
            tracking_result: TrackingResult from tracker
            detections: List of FusedTarget detections
        """
        # Update current data
        self.current_tracks = tracking_result.tracks if hasattr(tracking_result, 'tracks') else []
        self.current_detections = detections if detections else []
        
        # DEEP DEBUG - print everything
        print(f"\n=== VISUALIZER DEBUG ===")
        print(f"Received tracking_result type: {type(tracking_result)}")
        print(f"Has tracks attribute: {hasattr(tracking_result, 'tracks')}")
        if hasattr(tracking_result, 'tracks'):
            print(f"Tracks list: {tracking_result.tracks}")
            print(f"Number of tracks: {len(tracking_result.tracks)}")
        print(f"Current tracks assigned: {len(self.current_tracks)}")
        print(f"Current detections: {len(self.current_detections)}")
        
        for i, track in enumerate(self.current_tracks):
            print(f"Track {i}: id={getattr(track, 'id', 'NO_ID')}, pos={track.get_position()}, status={getattr(track, 'status', 'NO_STATUS')}")
        
        for i, det in enumerate(self.current_detections):
            print(f"Detection {i}: pos=({getattr(det, 'x', 'NO_X')}, {getattr(det, 'y', 'NO_Y')})")
        print(f"========================")
        
        
        # Update track histories
        self._update_track_histories()
        
        # Redraw everything
        self.setup_plot()
        
        # Plot detections
        if self.current_detections:
            det_x = [d.x for d in self.current_detections]
            det_y = [d.y for d in self.current_detections]
            self.ax.scatter(det_x, det_y, s=60, c='lightblue', 
                          edgecolors='blue', alpha=0.7, marker='o', label='Detections')
        
        # Plot track histories
        for track_id, history in self.track_histories.items():
            if len(history) > 1:
                positions = np.array([(x, y) for x, y, t in history])
                color = self.track_colors[track_id]
                self.ax.plot(positions[:, 0], positions[:, 1], '-', 
                           color=color, alpha=0.5, linewidth=2)
        
        # Plot current tracks
        for track in self.current_tracks:
            pos = track.get_position()
            color = self.track_colors.get(track.id, 'gray')
            
            # Determine track style based on status
            if track.status == TrackStatus.CONFIRMED:
                marker_size = 100
                edge_color = 'darkred'
                face_color = color
            elif track.status == TrackStatus.NEW:
                marker_size = 80
                edge_color = 'darkorange'
                face_color = 'orange'
            else:  # LOST
                marker_size = 60
                edge_color = 'gray'
                face_color = 'lightgray'
            
            # Plot track marker
            self.ax.scatter(pos[0], pos[1], s=marker_size, c=face_color, 
                          edgecolors=edge_color, linewidths=2, alpha=0.8)
            
            # Track ID label
            self.ax.annotate(f'{track.id}', (pos[0], pos[1]), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=edge_color)
            
            # Velocity vector for active tracks
            if track.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]:
                vel = track.get_velocity()
                vel_scale = 0.1  # Scale factor for velocity arrows
                if abs(vel[0]) > 0.01 or abs(vel[1]) > 0.01:  # Only show if moving
                    self.ax.arrow(pos[0], pos[1], vel[0]*vel_scale, vel[1]*vel_scale,
                                head_width=0.03, head_length=0.02, fc=color, ec=color,
                                alpha=0.8)
        
        # Update title with stats
        active_tracks = len([t for t in self.current_tracks if t.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]])
        confirmed_tracks = len([t for t in self.current_tracks if t.status == TrackStatus.CONFIRMED])
        self.ax.set_title(f'Live Tracking - Active: {active_tracks}, Confirmed: {confirmed_tracks}, Detections: {len(self.current_detections)}')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow',
                      markersize=8, markeredgecolor='black', label='Sensors'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=8, markeredgecolor='blue', label='Detections'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, markeredgecolor='darkred', label='Confirmed Tracks'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                      markersize=8, markeredgecolor='darkorange', label='New Tracks'),
            plt.Line2D([0], [0], linestyle='-', color='gray', alpha=0.7, label='Track History')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Force update
        plt.draw()
        plt.pause(0.001)
    
    def _update_track_histories(self):
        """Update track history storage."""
        current_track_ids = set()
        
        for track in self.current_tracks:
            track_id = track.id
            current_track_ids.add(track_id)
            
            # Initialize history for new tracks
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=50)
                self.track_colors[track_id] = self.color_cycle[self.color_index % len(self.color_cycle)]
                self.color_index += 1
            
            # Add current position to history
            pos = track.get_position()
            self.track_histories[track_id].append((pos[0], pos[1], time.time()))
        
        # Clean up old tracks
        old_tracks = set(self.track_histories.keys()) - current_track_ids
        for track_id in old_tracks:
            # Keep history for a bit after track is lost
            if len(self.track_histories[track_id]) > 0:
                last_time = self.track_histories[track_id][-1][2]
                if time.time() - last_time > 5.0:  # 5 seconds
                    del self.track_histories[track_id]
                    del self.track_colors[track_id]
    
    def close(self):
        """Close visualization."""
        plt.ioff()
        plt.close(self.fig)