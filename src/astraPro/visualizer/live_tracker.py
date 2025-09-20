"""
Live tracking visualization using matplotlib.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which is more stable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import queue
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import yaml

# Import TrackStatus for proper status checking
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from astraPro.aitracker.models import TrackStatus


class LiveTracker:
    """
    Real-time visualization of tracking data.
    """
    
    def __init__(self, field_size: Tuple[float, float] = (3.0, 3.0), 
                 config_file: str = "config.yaml",
                 history_length: int = 50):
        """
        Initialize live tracker.
        
        Args:
            field_size: (width, height) of tracking field in meters
            config_file: Path to sensor configuration file
            history_length: Number of track history points to display
        """
        self.field_size = field_size
        self.history_length = history_length
        
        # Load sensor positions
        self.sensor_positions = self._load_sensor_config(config_file)
        
        # Data queues for thread-safe updates
        self.tracking_queue = queue.Queue()
        self.detection_queue = queue.Queue()
        
        # Track history storage
        self.track_histories: Dict[int, deque] = {}
        self.track_colors: Dict[int, str] = {}
        self.color_cycle = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.color_index = 0
        
        # Current data
        self.current_tracks = []
        self.current_detections = []
        
        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()
        
        # Animation
        self.animation = None
        self.running = False
    
    def _load_sensor_config(self, config_file: str) -> List[Tuple[float, float]]:
        """Load sensor positions from config file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                sensors = config.get('sensors', [])
                return [(s['position']['x'], s['position']['y']) for s in sensors]
        except (FileNotFoundError, KeyError):
            # Default sensor positions matching simulator
            return [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5), (0.0, 1.5)]
    
    def setup_plot(self):
        """Setup the matplotlib plot."""
        self.ax.set_xlim(-self.field_size[0]/2, self.field_size[0]/2)
        self.ax.set_ylim(-self.field_size[1]/2, self.field_size[1]/2)
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
        
        # Draw sensors
        for i, (x, y) in enumerate(self.sensor_positions):
            self.ax.plot(x, y, 's', markersize=8, color='black', 
                        markerfacecolor='yellow', markeredgewidth=2)
            self.ax.annotate(f'S{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # Create legend elements
        self.legend_elements = []
        self.update_legend()
    
    def update_legend(self):
        """Update plot legend."""
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
    
    def add_tracking_data(self, tracking_result, detections):
        """
        Add new tracking data (thread-safe).
        
        Args:
            tracking_result: TrackingResult from tracker
            detections: List of FusedTarget detections
        """
        try:
            self.tracking_queue.put((tracking_result, detections), block=False)
        except queue.Full:
            pass  # Skip if queue is full
    
    def _update_data(self):
        """Update data from queues."""
        # Process all available data
        while not self.tracking_queue.empty():
            try:
                tracking_result, detections = self.tracking_queue.get_nowait()
                self.current_tracks = tracking_result.tracks
                self.current_detections = detections
                
                
                self._update_track_histories()
            except queue.Empty:
                break
    
    def _update_track_histories(self):
        """Update track history storage."""
        current_track_ids = set()
        
        for track in self.current_tracks:
            track_id = track.id
            current_track_ids.add(track_id)
            
            # Initialize history for new tracks
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.history_length)
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
    
    def _animate(self, frame):
        """Animation update function."""
        if not self.running:
            return []
        
        # Update data from queue
        self._update_data()
        
        # Only clear and redraw if we have new data
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = 0
        
        current_time = time.time()
        if current_time - self._last_update_time < 0.05:  # Limit to 20 FPS max
            return []
        
        self._last_update_time = current_time
        
        # Clear previous plots (except static elements)
        self.ax.clear()
        self.setup_plot()
        
        # Plot detections with scatter for better visibility
        if self.current_detections:
            det_x = [d.x for d in self.current_detections]
            det_y = [d.y for d in self.current_detections]
            self.ax.scatter(det_x, det_y, s=50, c='lightblue', 
                          edgecolors='blue', alpha=0.7, marker='o')
        
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
            vel = track.get_velocity()
            color = self.track_colors.get(track.id, 'gray')
            
            # Track marker
            if track.status == TrackStatus.CONFIRMED:
                marker_size = 10
                edge_color = 'darkred'
                face_color = color
            elif track.status == TrackStatus.NEW:
                marker_size = 8
                edge_color = 'darkorange'
                face_color = 'orange'
            else:  # lost
                marker_size = 6
                edge_color = 'gray'
                face_color = 'lightgray'
            
            # Use scatter instead of plot for better visibility
            self.ax.scatter(pos[0], pos[1], s=marker_size*10, c=face_color, 
                          edgecolors=edge_color, linewidths=2, alpha=0.8)
            
            # Track ID label
            self.ax.annotate(f'{track.id}', (pos[0], pos[1]), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=edge_color)
            
            # Velocity vector
            if track.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]:
                vel_scale = 0.1  # Scale factor for velocity arrows
                self.ax.arrow(pos[0], pos[1], vel[0]*vel_scale, vel[1]*vel_scale,
                            head_width=0.03, head_length=0.02, fc=color, ec=color,
                            alpha=0.8)
        
        # Update title with stats
        active_tracks = len([t for t in self.current_tracks if t.status in [TrackStatus.CONFIRMED, TrackStatus.NEW]])
        confirmed_tracks = len([t for t in self.current_tracks if t.status == TrackStatus.CONFIRMED])
        self.ax.set_title(f'Live Tracking - Active: {active_tracks}, Confirmed: {confirmed_tracks}, Detections: {len(self.current_detections)}')
        
        # Force redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def start(self, update_interval: int = 100):
        """
        Start live visualization.
        
        Args:
            update_interval: Update interval in milliseconds
        """
        self.running = True
        self.animation = FuncAnimation(self.fig, self._animate, 
                                     interval=update_interval, blit=False,
                                     cache_frame_data=False)
        try:
            plt.show(block=False)  # Non-blocking show
            plt.pause(0.001)  # Small pause to ensure window opens
        except Exception as e:
            print(f"Warning: Could not open visualization window: {e}")
            self.running = False
    
    def stop(self):
        """Stop visualization."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
    
    def save_frame(self, filename: str):
        """Save current frame to file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Frame saved to {filename}")


class LiveTrackerThread:
    """
    Threaded version of live tracker for non-blocking operation.
    """
    
    def __init__(self, **kwargs):
        self.tracker = LiveTracker(**kwargs)
        self.thread = None
        self.running = False
    
    def start(self):
        """Start visualization in separate thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            print("Live tracker started in background thread")
    
    def _run(self):
        """Run visualization."""
        plt.ion()  # Turn on interactive mode
        self.tracker.start()
        
        # Keep updating until stopped
        while self.running:
            plt.pause(0.1)
    
    def add_data(self, tracking_result, detections):
        """Add tracking data."""
        self.tracker.add_tracking_data(tracking_result, detections)
    
    def stop(self):
        """Stop visualization."""
        self.running = False
        self.tracker.stop()
        plt.ioff()  # Turn off interactive mode
        if self.thread:
            self.thread.join(timeout=1.0)