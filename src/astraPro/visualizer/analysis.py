"""
Analysis tools for tracking data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive plots will be disabled.")


class TrackingAnalyzer:
    """
    Analysis tools for CSV tracking data.
    """
    
    def __init__(self, data_dir: str, session_name: str):
        """
        Initialize analyzer with data files.
        
        Args:
            data_dir: Directory containing CSV files
            session_name: Name of the tracking session
        """
        self.data_dir = data_dir
        self.session_name = session_name
        
        # File paths
        self.tracks_file = os.path.join(data_dir, f"{session_name}_tracks.csv")
        self.detections_file = os.path.join(data_dir, f"{session_name}_detections.csv")
        self.stats_file = os.path.join(data_dir, f"{session_name}_stats.csv")
        self.ground_truth_file = os.path.join(data_dir, f"{session_name}_ground_truth.csv")
        
        # Load data
        self.tracks_df = None
        self.detections_df = None
        self.stats_df = None
        self.ground_truth_df = None
        self.load_data()
    
    def load_data(self):
        """Load CSV data into DataFrames."""
        try:
            if os.path.exists(self.tracks_file):
                self.tracks_df = pd.read_csv(self.tracks_file)
                print(f"Loaded {len(self.tracks_df)} track records")
            
            if os.path.exists(self.detections_file):
                self.detections_df = pd.read_csv(self.detections_file)
                print(f"Loaded {len(self.detections_df)} detection records")
            
            if os.path.exists(self.stats_file):
                self.stats_df = pd.read_csv(self.stats_file)
                print(f"Loaded {len(self.stats_df)} statistics records")
            
            if os.path.exists(self.ground_truth_file):
                self.ground_truth_df = pd.read_csv(self.ground_truth_file)
                print(f"Loaded {len(self.ground_truth_df)} ground truth records")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def track_summary(self) -> Dict:
        """Generate summary statistics for tracks."""
        if self.tracks_df is None:
            return {}
        
        summary = {}
        
        # Basic stats
        summary['total_tracks'] = self.tracks_df['track_id'].nunique()
        summary['total_records'] = len(self.tracks_df)
        summary['session_duration'] = self.tracks_df['timestamp'].max() - self.tracks_df['timestamp'].min()
        
        # Track lifecycles
        track_groups = self.tracks_df.groupby('track_id')
        summary['track_lifetimes'] = track_groups['timestamp'].apply(lambda x: x.max() - x.min())
        summary['avg_track_lifetime'] = summary['track_lifetimes'].mean()
        summary['max_track_lifetime'] = summary['track_lifetimes'].max()
        
        # Track quality
        summary['avg_confidence'] = self.tracks_df['confidence'].mean()
        summary['confirmed_tracks'] = len(self.tracks_df[self.tracks_df['status'] == 'confirmed']['track_id'].unique())
        
        # Movement analysis
        summary['avg_speed'] = np.sqrt(self.tracks_df['vel_x']**2 + self.tracks_df['vel_y']**2).mean()
        summary['max_speed'] = np.sqrt(self.tracks_df['vel_x']**2 + self.tracks_df['vel_y']**2).max()
        
        return summary
    
    def plot_track_trajectories(self, track_ids: Optional[List[int]] = None, 
                               save_path: Optional[str] = None):
        """
        Plot track trajectories.
        
        Args:
            track_ids: Specific track IDs to plot (all if None)
            save_path: Path to save plot (display if None)
        """
        if self.tracks_df is None:
            print("No track data available")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Filter tracks if specified
        if track_ids is not None:
            df = self.tracks_df[self.tracks_df['track_id'].isin(track_ids)]
        else:
            df = self.tracks_df
        
        # Plot each track
        unique_tracks = df['track_id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tracks)))
        
        for i, track_id in enumerate(unique_tracks):
            track_data = df[df['track_id'] == track_id].sort_values('timestamp')
            
            # Plot trajectory
            plt.plot(track_data['pos_x'], track_data['pos_y'], 
                    color=colors[i], linewidth=2, alpha=0.7, 
                    label=f'Track {track_id}')
            
            # Mark start and end
            plt.plot(track_data['pos_x'].iloc[0], track_data['pos_y'].iloc[0], 
                    'o', color=colors[i], markersize=8, markeredgecolor='black')
            plt.plot(track_data['pos_x'].iloc[-1], track_data['pos_y'].iloc[-1], 
                    's', color=colors[i], markersize=8, markeredgecolor='black')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Track Trajectories')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_speed_analysis(self, save_path: Optional[str] = None):
        """Plot speed analysis."""
        if self.tracks_df is None:
            print("No track data available")
            return
        
        # Calculate speeds
        speeds = np.sqrt(self.tracks_df['vel_x']**2 + self.tracks_df['vel_y']**2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Speed histogram
        axes[0, 0].hist(speeds, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Speed (m/s)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Speed Distribution')
        
        # Speed over time
        axes[0, 1].scatter(self.tracks_df['timestamp'], speeds, alpha=0.5)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Speed (m/s)')
        axes[0, 1].set_title('Speed vs Time')
        
        # Speed by track
        track_speeds = self.tracks_df.groupby('track_id').apply(
            lambda x: np.sqrt(x['vel_x']**2 + x['vel_y']**2).mean()
        )
        axes[1, 0].bar(range(len(track_speeds)), track_speeds.values)
        axes[1, 0].set_xlabel('Track ID')
        axes[1, 0].set_ylabel('Average Speed (m/s)')
        axes[1, 0].set_title('Average Speed by Track')
        axes[1, 0].set_xticks(range(len(track_speeds)))
        axes[1, 0].set_xticklabels(track_speeds.index)
        
        # Velocity components
        axes[1, 1].scatter(self.tracks_df['vel_x'], self.tracks_df['vel_y'], alpha=0.5)
        axes[1, 1].set_xlabel('X Velocity (m/s)')
        axes[1, 1].set_ylabel('Y Velocity (m/s)')
        axes[1, 1].set_title('Velocity Components')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Speed analysis saved to {save_path}")
        else:
            plt.show()
    
    def plot_tracking_performance(self, save_path: Optional[str] = None):
        """Plot tracking system performance metrics."""
        if self.stats_df is None:
            print("No statistics data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Track counts over time
        time_rel = self.stats_df['timestamp'] - self.stats_df['timestamp'].min()
        axes[0, 0].plot(time_rel, self.stats_df['total_tracks'], label='Total', linewidth=2)
        axes[0, 0].plot(time_rel, self.stats_df['confirmed_tracks'], label='Confirmed', linewidth=2)
        axes[0, 0].plot(time_rel, self.stats_df['new_tracks'], label='New', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Track Count')
        axes[0, 0].set_title('Track Counts Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Detection count over time
        axes[0, 1].plot(time_rel, self.stats_df['num_detections'], color='orange', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Detection Count')
        axes[0, 1].set_title('Detections Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average confidence over time
        axes[1, 0].plot(time_rel, self.stats_df['avg_confidence'], color='green', linewidth=2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].set_title('Tracking Confidence Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Detection vs track efficiency
        efficiency = self.stats_df['confirmed_tracks'] / (self.stats_df['num_detections'] + 1e-6)
        axes[1, 1].plot(time_rel, efficiency, color='red', linewidth=2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Tracks/Detections Ratio')
        axes[1, 1].set_title('Tracking Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance analysis saved to {save_path}")
        else:
            plt.show()
    
    def export_report(self, output_file: str):
        """Export comprehensive analysis report."""
        summary = self.track_summary()
        
        # Create output directory
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        base_name = os.path.splitext(output_file)[0]
        
        self.plot_track_trajectories(save_path=f"{base_name}_trajectories.png")
        self.plot_speed_analysis(save_path=f"{base_name}_speed_analysis.png")
        self.plot_tracking_performance(save_path=f"{base_name}_performance.png")
        
        # Generate text report
        with open(output_file, 'w') as f:
            f.write(f"# Tracking Analysis Report\n")
            f.write(f"Session: {self.session_name}\n\n")
            
            f.write("## Summary Statistics\n")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value:.3f}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            
            f.write("\n## Track Details\n")
            if self.tracks_df is not None:
                track_summary = self.tracks_df.groupby('track_id').agg({
                    'timestamp': ['min', 'max', 'count'],
                    'confidence': 'mean',
                    'hits': 'max'
                }).round(3)
                f.write(track_summary.to_string())
        
        print(f"Analysis report exported to {output_file}")
    
    def calculate_tracking_scorecard(self, max_distance: float = 0.5) -> Dict:
        """
        Calculate tracking performance scorecard against ground truth.
        
        Args:
            max_distance: Maximum distance to consider a track as correctly associated
            
        Returns:
            Dictionary containing performance metrics
        """
        if self.tracks_df is None or self.ground_truth_df is None:
            print("Cannot calculate scorecard: missing tracks or ground truth data")
            return {}
        
        scorecard = {}
        
        # Get confirmed tracks only for scoring
        confirmed_tracks = self.tracks_df[self.tracks_df['status'] == 'confirmed']
        
        # If no confirmed tracks, use all tracks for analysis
        if len(confirmed_tracks) == 0:
            print("No confirmed tracks found, analyzing all tracks")
            confirmed_tracks = self.tracks_df
        
        # Find approximately common timestamps (within tolerance)
        tolerance = 0.1  # 100ms tolerance
        track_timestamps = sorted(confirmed_tracks['timestamp'].unique())
        gt_timestamps = sorted(self.ground_truth_df['timestamp'].unique())
        
        common_timestamps = []
        for track_ts in track_timestamps:
            # Find closest ground truth timestamp within tolerance
            closest_gt_ts = min(gt_timestamps, key=lambda x: abs(x - track_ts))
            if abs(closest_gt_ts - track_ts) <= tolerance:
                common_timestamps.append(track_ts)
        
        if not common_timestamps:
            print("No common timestamps between tracks and ground truth")
            return {}
        
        # Calculate metrics at each timestamp
        total_matches = 0
        total_false_positives = 0
        total_false_negatives = 0
        position_errors = []
        velocity_errors = []
        
        # Association tracking
        track_associations = {}  # track_id -> object_id mapping
        
        for timestamp in common_timestamps:
            # Get tracks and ground truth at this timestamp
            tracks_at_t = confirmed_tracks[confirmed_tracks['timestamp'] == timestamp]
            
            # Find closest ground truth timestamp
            closest_gt_ts = min(gt_timestamps, key=lambda x: abs(x - timestamp))
            gt_at_t = self.ground_truth_df[self.ground_truth_df['timestamp'] == closest_gt_ts]
            
            # Calculate distance matrix between tracks and ground truth objects
            matches = []
            used_tracks = set()
            used_objects = set()
            
            for _, track in tracks_at_t.iterrows():
                best_match = None
                best_distance = float('inf')
                
                for _, gt_obj in gt_at_t.iterrows():
                    if gt_obj['object_id'] in used_objects:
                        continue
                        
                    distance = np.sqrt((track['pos_x'] - gt_obj['pos_x'])**2 + 
                                     (track['pos_y'] - gt_obj['pos_y'])**2)
                    
                    if distance < max_distance and distance < best_distance:
                        best_match = gt_obj
                        best_distance = distance
                
                if best_match is not None:
                    matches.append({
                        'track_id': track['track_id'],
                        'object_id': best_match['object_id'],
                        'distance': best_distance,
                        'track': track,
                        'gt': best_match
                    })
                    used_tracks.add(track['track_id'])
                    used_objects.add(best_match['object_id'])
            
            # Count matches and errors
            total_matches += len(matches)
            total_false_positives += len(tracks_at_t) - len(matches)
            total_false_negatives += len(gt_at_t) - len(matches)
            
            # Calculate position and velocity errors for matches
            for match in matches:
                track = match['track']
                gt = match['gt']
                
                # Position error
                pos_error = np.sqrt((track['pos_x'] - gt['pos_x'])**2 + 
                                  (track['pos_y'] - gt['pos_y'])**2)
                position_errors.append(pos_error)
                
                # Velocity error
                vel_error = np.sqrt((track['vel_x'] - gt['vel_x'])**2 + 
                                  (track['vel_y'] - gt['vel_y'])**2)
                velocity_errors.append(vel_error)
                
                # Update track associations
                track_associations[track['track_id']] = gt['object_id']
        
        # Calculate overall metrics
        total_gt_objects = len(common_timestamps) * len(self.ground_truth_df['object_id'].unique())
        total_tracks = len(common_timestamps) * len(confirmed_tracks['track_id'].unique())
        
        # Traditional association-based precision and recall
        association_precision = total_matches / max(total_matches + total_false_positives, 1)
        association_recall = total_matches / max(total_matches + total_false_negatives, 1)
        
        # Quality-weighted precision: penalize for position and velocity errors
        if position_errors and velocity_errors:
            mean_pos_error = np.mean(position_errors)
            mean_vel_error = np.mean(velocity_errors)
            
            # Define acceptable error thresholds
            acceptable_pos_error = 0.1  # 10cm position error threshold
            acceptable_vel_error = 0.1  # 0.1 m/s velocity error threshold
            
            # Calculate quality scores (1.0 = perfect, 0.0 = unacceptable)
            pos_quality = max(0.0, 1.0 - (mean_pos_error / acceptable_pos_error))
            vel_quality = max(0.0, 1.0 - (mean_vel_error / acceptable_vel_error))
            
            # Combined quality score (average of position and velocity quality)
            tracking_quality = (pos_quality + vel_quality) / 2.0
            
            # Quality-weighted precision combines association success with tracking accuracy
            precision = association_precision * tracking_quality
        else:
            # Fallback to association-based precision if no error data
            precision = association_precision
            
        recall = association_recall
        f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
        
        # Track consistency (how often same track is associated with same object)
        track_consistency = self._calculate_track_consistency(track_associations)
        
        # Expected vs actual track count
        expected_tracks = len(self.ground_truth_df['object_id'].unique())
        actual_confirmed_tracks = len(confirmed_tracks['track_id'].unique())
        
        scorecard.update({
            'precision': precision,
            'association_precision': association_precision,  # Raw association success rate
            'recall': recall,
            'f1_score': f1_score,
            'track_consistency': track_consistency,
            'total_matches': total_matches,
            'false_positives': total_false_positives,
            'false_negatives': total_false_negatives,
            'expected_tracks': expected_tracks,
            'actual_confirmed_tracks': actual_confirmed_tracks,
            'track_count_accuracy': min(actual_confirmed_tracks / max(expected_tracks, 1), 1.0),
            'mean_position_error': np.mean(position_errors) if position_errors else float('inf'),
            'std_position_error': np.std(position_errors) if position_errors else float('inf'),
            'mean_velocity_error': np.mean(velocity_errors) if velocity_errors else float('inf'),
            'std_velocity_error': np.std(velocity_errors) if velocity_errors else float('inf'),
            'total_timestamps_analyzed': len(common_timestamps)
        })
        
        return scorecard
    
    def _calculate_track_consistency(self, track_associations: Dict) -> float:
        """Calculate how consistently tracks are associated with the same object."""
        if not track_associations:
            return 0.0
        
        # Count how many times each track was associated with each object
        track_object_counts = {}
        for track_id, object_id in track_associations.items():
            if track_id not in track_object_counts:
                track_object_counts[track_id] = {}
            if object_id not in track_object_counts[track_id]:
                track_object_counts[track_id][object_id] = 0
            track_object_counts[track_id][object_id] += 1
        
        # Calculate consistency score for each track
        consistencies = []
        for track_id, object_counts in track_object_counts.items():
            total_associations = sum(object_counts.values())
            max_associations = max(object_counts.values())
            consistency = max_associations / total_associations
            consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def plot_tracking_scorecard(self, save_path: Optional[str] = None):
        """Plot tracking performance scorecard."""
        scorecard = self.calculate_tracking_scorecard()
        
        if not scorecard:
            print("Cannot generate scorecard plot: no scorecard data")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Performance metrics
        metrics = ['precision', 'recall', 'f1_score', 'track_consistency', 'track_count_accuracy']
        values = [scorecard.get(m, 0) for m in metrics]
        colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values]
        
        axes[0, 0].bar(metrics, values, color=colors)
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Score (0-1)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Position errors
        if self.tracks_df is not None and self.ground_truth_df is not None:
            # Plot ground truth vs tracks trajectories
            axes[0, 1].set_title('Ground Truth vs Tracks')
            
            # Plot ground truth
            for obj_id in self.ground_truth_df['object_id'].unique():
                gt_obj = self.ground_truth_df[self.ground_truth_df['object_id'] == obj_id]
                axes[0, 1].plot(gt_obj['pos_x'], gt_obj['pos_y'], 
                              'o-', label=f'GT Object {obj_id}', alpha=0.7)
            
            # Plot tracks (confirmed if available, otherwise all)
            plot_tracks = self.tracks_df[self.tracks_df['status'] == 'confirmed']
            if len(plot_tracks) == 0:
                plot_tracks = self.tracks_df
            for track_id in plot_tracks['track_id'].unique():
                track = plot_tracks[plot_tracks['track_id'] == track_id]
                axes[0, 1].plot(track['pos_x'], track['pos_y'], 
                              's--', label=f'Track {track_id}', alpha=0.7)
            
            axes[0, 1].set_xlabel('X Position (m)')
            axes[0, 1].set_ylabel('Y Position (m)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 2].text(0.1, 0.8, f"Mean Position Error: {scorecard.get('mean_position_error', 0):.3f} m", 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].text(0.1, 0.7, f"Std Position Error: {scorecard.get('std_position_error', 0):.3f} m", 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].text(0.1, 0.6, f"Mean Velocity Error: {scorecard.get('mean_velocity_error', 0):.3f} m/s", 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].text(0.1, 0.5, f"Expected Tracks: {scorecard.get('expected_tracks', 0)}", 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].text(0.1, 0.4, f"Actual Tracks: {scorecard.get('actual_confirmed_tracks', 0)}", 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].set_title('Error Statistics')
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].axis('off')
        
        # Confusion matrix-style visualization
        axes[1, 0].bar(['True Positives', 'False Positives', 'False Negatives'], 
                      [scorecard.get('total_matches', 0), 
                       scorecard.get('false_positives', 0), 
                       scorecard.get('false_negatives', 0)],
                      color=['green', 'orange', 'red'])
        axes[1, 0].set_title('Detection Results')
        axes[1, 0].set_ylabel('Count')
        
        # Overall grade
        overall_score = (scorecard.get('f1_score', 0) + 
                        scorecard.get('track_consistency', 0) + 
                        scorecard.get('track_count_accuracy', 0)) / 3
        
        grade_color = 'green' if overall_score >= 0.8 else 'orange' if overall_score >= 0.6 else 'red'
        grade_text = 'A' if overall_score >= 0.9 else 'B' if overall_score >= 0.8 else 'C' if overall_score >= 0.7 else 'D' if overall_score >= 0.6 else 'F'
        
        axes[1, 1].text(0.5, 0.6, f'Overall Grade', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=16, weight='bold')
        axes[1, 1].text(0.5, 0.4, f'{grade_text}', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=48, weight='bold', color=grade_color)
        axes[1, 1].text(0.5, 0.2, f'Score: {overall_score:.3f}', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        # Performance over time
        if self.stats_df is not None:
            time_rel = self.stats_df['timestamp'] - self.stats_df['timestamp'].min()
            efficiency = self.stats_df['confirmed_tracks'] / (self.stats_df['num_detections'] + 1e-6)
            axes[1, 2].plot(time_rel, efficiency, linewidth=2)
            axes[1, 2].axhline(y=scorecard.get('expected_tracks', 3)/10, color='red', linestyle='--', 
                             label='Expected Efficiency')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Tracks/Detections Ratio')
            axes[1, 2].set_title('Tracking Efficiency Over Time')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Scorecard plot saved to {save_path}")
        else:
            plt.show()
        
        return scorecard
    
    def plot_interactive_trajectories(self, save_path: Optional[str] = None, 
                                    include_measurements: bool = True,
                                    include_ground_truth: bool = True) -> str:
        """
        Create interactive HTML trajectory plot with measurements and ground truth.
        
        Args:
            save_path: Path to save HTML file (auto-generated if None)
            include_measurements: Whether to show measurement dots
            include_ground_truth: Whether to show ground truth if available
            
        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive plot.")
            return ""
        
        if self.tracks_df is None:
            print("No track data available")
            return ""
        
        # Generate save path if not provided
        if save_path is None:
            save_path = f"{self.session_name}_interactive_trajectories.html"
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Color palette for tracks
        colors = px.colors.qualitative.Set1
        
        # Plot tracks (use all tracks if no confirmed ones)
        confirmed_tracks = self.tracks_df[self.tracks_df['status'] == 'confirmed']
        if len(confirmed_tracks) == 0:
            print("No confirmed tracks for interactive plot, using all tracks")
            confirmed_tracks = self.tracks_df
        unique_tracks = confirmed_tracks['track_id'].unique()
        
        for i, track_id in enumerate(unique_tracks):
            track_data = confirmed_tracks[confirmed_tracks['track_id'] == track_id].sort_values('timestamp')
            color = colors[i % len(colors)]
            
            # Track trajectory line
            fig.add_trace(go.Scatter(
                x=track_data['pos_x'],
                y=track_data['pos_y'],
                mode='lines+markers',
                name=f'Track {track_id}',
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
                hovertemplate=(
                    f'<b>Track {track_id}</b><br>' +
                    'Position: (%{x:.2f}, %{y:.2f})<br>' +
                    'Velocity: (%{customdata[0]:.2f}, %{customdata[1]:.2f})<br>' +
                    'Confidence: %{customdata[2]:.2f}<br>' +
                    'Hits: %{customdata[3]}<br>' +
                    'Time: %{customdata[4]:.2f}s<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    track_data['vel_x'],
                    track_data['vel_y'], 
                    track_data['confidence'],
                    track_data['hits'],
                    track_data['timestamp'] - track_data['timestamp'].min()
                )),
                legendgroup=f'track_{track_id}'
            ))
            
            # Start and end markers
            fig.add_trace(go.Scatter(
                x=[track_data['pos_x'].iloc[0]],
                y=[track_data['pos_y'].iloc[0]],
                mode='markers',
                name=f'Track {track_id} Start',
                marker=dict(
                    size=12, 
                    color=color,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                showlegend=False,
                hovertemplate=f'<b>Track {track_id} START</b><br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                legendgroup=f'track_{track_id}'
            ))
            
            fig.add_trace(go.Scatter(
                x=[track_data['pos_x'].iloc[-1]],
                y=[track_data['pos_y'].iloc[-1]],
                mode='markers',
                name=f'Track {track_id} End',
                marker=dict(
                    size=12, 
                    color=color,
                    symbol='square',
                    line=dict(color='black', width=2)
                ),
                showlegend=False,
                hovertemplate=f'<b>Track {track_id} END</b><br>Position: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>',
                legendgroup=f'track_{track_id}'
            ))
        
        # Add detection measurements if available and requested
        if include_measurements and self.detections_df is not None:
            fig.add_trace(go.Scatter(
                x=self.detections_df['pos_x'],
                y=self.detections_df['pos_y'],
                mode='markers',
                name='Detections',
                marker=dict(
                    size=8,
                    color='lightblue',
                    symbol='diamond',
                    opacity=0.6,
                    line=dict(color='blue', width=1)
                ),
                hovertemplate=(
                    '<b>Detection</b><br>' +
                    'Position: (%{x:.2f}, %{y:.2f})<br>' +
                    'Confidence: %{customdata[0]:.2f}<br>' +
                    'Sensors: %{customdata[1]}<br>' +
                    'Time: %{customdata[2]:.2f}s<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    self.detections_df['confidence'],
                    self.detections_df['num_sensors'],
                    self.detections_df['timestamp'] - self.detections_df['timestamp'].min()
                ))
            ))
        
        # Add ground truth if available and requested
        if include_ground_truth and self.ground_truth_df is not None:
            gt_colors = ['red', 'orange', 'purple']  # Different colors for GT objects
            for obj_id in self.ground_truth_df['object_id'].unique():
                gt_obj = self.ground_truth_df[self.ground_truth_df['object_id'] == obj_id].sort_values('timestamp')
                color = gt_colors[obj_id % len(gt_colors)]
                
                fig.add_trace(go.Scatter(
                    x=gt_obj['pos_x'],
                    y=gt_obj['pos_y'],
                    mode='lines+markers',
                    name=f'Ground Truth Object {obj_id}',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=4, color=color, symbol='star'),
                    hovertemplate=(
                        f'<b>Ground Truth Object {obj_id}</b><br>' +
                        'Position: (%{x:.2f}, %{y:.2f})<br>' +
                        'Velocity: (%{customdata[0]:.2f}, %{customdata[1]:.2f})<br>' +
                        'Time: %{customdata[2]:.2f}s<br>' +
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        gt_obj['vel_x'],
                        gt_obj['vel_y'],
                        gt_obj['timestamp'] - gt_obj['timestamp'].min()
                    ))
                ))
        
        # Add sensor positions (fixed positions at field corners)
        sensor_positions = [
            (-1.5, -1.5, 'SW Sensor'),
            (1.5, -1.5, 'SE Sensor'),
            (1.5, 1.5, 'NE Sensor'),
            (-1.5, 1.5, 'NW Sensor')
        ]
        
        for x, y, name in sensor_positions:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                name=name,
                marker=dict(size=15, color='black', symbol='triangle-up'),
                hovertemplate=f'<b>{name}</b><br>Position: ({x:.1f}, {y:.1f})<extra></extra>'
            ))
        
        # Update layout for better interactivity
        fig.update_layout(
            title=dict(
                text=f'Interactive Trajectory Analysis - {self.session_name}',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title='X Position (m)',
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                title='Y Position (m)',
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                scaleanchor='x',  # Keep aspect ratio 1:1
                scaleratio=1
            ),
            plot_bgcolor='white',
            width=1000,
            height=800,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        # Add field boundary (3m x 3m field)
        field_boundary_x = [-1.5, 1.5, 1.5, -1.5, -1.5]
        field_boundary_y = [-1.5, -1.5, 1.5, 1.5, -1.5]
        fig.add_trace(go.Scatter(
            x=field_boundary_x,
            y=field_boundary_y,
            mode='lines',
            name='Field Boundary',
            line=dict(color='gray', width=2, dash='dot'),
            hovertemplate='<b>Field Boundary</b><extra></extra>'
        ))
        
        # Add custom controls with HTML/JavaScript
        html_controls = self._generate_filter_controls(unique_tracks, 
                                                     self.detections_df is not None,
                                                     self.ground_truth_df is not None)
        
        # Save as HTML with custom controls - using simpler approach
        config = {'displayModeBar': True, 'displaylogo': False}
        
        # First try the simple plotly offline approach
        try:
            # Generate plotly HTML
            fig_html = pyo.plot(fig, output_type='div', config=config, include_plotlyjs='cdn')
            
            # Generate the complete HTML with embedded plot
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Interactive Trajectory Analysis - {self.session_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .controls {{ 
                        background: #f5f5f5; 
                        padding: 15px; 
                        border-radius: 5px; 
                        margin-bottom: 15px;
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        align-items: center;
                    }}
                    .control-group {{
                        display: flex;
                        flex-direction: column;
                        align-items: flex-start;
                    }}
                    .control-group label {{
                        font-weight: bold;
                        margin-bottom: 5px;
                    }}
                    select, input[type="range"] {{
                        padding: 5px;
                        border: 1px solid #ccc;
                        border-radius: 3px;
                    }}
                    .checkbox-group {{
                        display: flex;
                        gap: 15px;
                        align-items: center;
                    }}
                    .checkbox-item {{
                        display: flex;
                        align-items: center;
                        gap: 5px;
                    }}
                    .plotly-graph-div {{ margin: 0 !important; }}
                    .time-info {{ 
                        font-size: 12px; 
                        color: #666; 
                        margin-top: 5px; 
                    }}
                    .filter-info {{
                        background: #e3f2fd;
                        padding: 10px;
                        border-radius: 5px;
                        margin-top: 10px;
                        font-size: 14px;
                    }}
                </style>
            </head>
            <body>
                <h1>Interactive Trajectory Analysis - {self.session_name}</h1>
                
                {html_controls}
                
                {fig_html}
                
                <script>
                    {self._generate_filter_javascript(unique_tracks)}
                    
                    // Initialize when page loads
                    window.addEventListener('load', function() {{
                        console.log('Page loaded, initializing filters...');
                        setTimeout(function() {{
                            testFilters();
                            // Don't auto-apply filters yet - let user control them
                        }}, 1000);
                    }});
                    
                    // Add a button for testing (temporary)
                    setTimeout(function() {{
                        var testButton = document.createElement('button');
                        testButton.innerHTML = 'Test Filters (Debug)';
                        testButton.onclick = testFilters;
                        testButton.style.cssText = 'margin: 10px; padding: 5px; background: orange; color: white; border: none;';
                        document.body.insertBefore(testButton, document.querySelector('.filter-info'));
                    }}, 500);
                </script>
                
            </body>
            </html>
            """
            
            with open(save_path, 'w') as f:
                f.write(full_html)
                
        except Exception as e:
            print(f"Error generating advanced interactive plot: {e}")
            print("Falling back to basic plotly export...")
            
            # Fallback: Just export the basic plotly HTML
            basic_html = pyo.plot(fig, filename=save_path, auto_open=False, config=config)
            
        print(f"Interactive trajectory plot saved to {save_path}")
        return save_path
    
    def _generate_filter_controls(self, track_ids, has_detections, has_ground_truth):
        """Generate HTML for filter controls."""
        track_options = ""
        for track_id in sorted(track_ids):
            track_options += f'<option value="{track_id}">Track {track_id}</option>'
        
        detection_checkbox = ""
        if has_detections:
            detection_checkbox = '''
                <div class="checkbox-item">
                    <input type="checkbox" id="showDetections" checked>
                    <label for="showDetections">Show Detections</label>
                </div>
            '''
        
        ground_truth_checkbox = ""
        if has_ground_truth:
            ground_truth_checkbox = '''
                <div class="checkbox-item">
                    <input type="checkbox" id="showGroundTruth" checked>
                    <label for="showGroundTruth">Show Ground Truth</label>
                </div>
            '''
        
        return f'''
        <div class="controls">
            <div class="control-group">
                <label for="trackFilter">Track Filter:</label>
                <select id="trackFilter" onchange="updateTrackVisibility()">
                    <option value="all">Show All Tracks</option>
                    {track_options}
                </select>
            </div>
            
            <div class="control-group">
                <label>Display Options:</label>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="showTracks" checked onchange="updateElementVisibility()">
                        <label for="showTracks">Show Tracks</label>
                    </div>
                    {detection_checkbox}
                    {ground_truth_checkbox}
                    <div class="checkbox-item">
                        <input type="checkbox" id="showSensors" checked onchange="updateElementVisibility()">
                        <label for="showSensors">Show Sensors</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="showBoundary" checked onchange="updateElementVisibility()">
                        <label for="showBoundary">Show Field Boundary</label>
                    </div>
                </div>
            </div>
            
            <div class="control-group">
                <label for="timeRange">Time Filter:</label>
                <input type="range" id="timeRange" min="0" max="100" value="100" 
                       onchange="updateTimeFilter()" oninput="updateTimeDisplay()">
                <div class="time-info" id="timeInfo">Showing: Full Duration</div>
            </div>
            
            <div class="control-group">
                <button onclick="resetFilters()" style="padding: 8px 15px; background: #2196F3; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    Reset All Filters
                </button>
            </div>
        </div>
        
        <div class="filter-info">
            <strong>Filter Tips:</strong>
            • Select a specific track to focus on one trajectory
            • Use checkboxes to hide/show different element types
            • Drag the time slider to see tracking evolution over time
            • Click legend items to toggle individual traces
            • Use mouse wheel to zoom, drag to pan
        </div>
        '''
    
    def _generate_filter_javascript(self, track_ids):
        """Generate JavaScript for interactive filtering."""
        track_ids_js = str(list(track_ids)).replace("'", '"')
        
        return f'''
        var allTrackIds = {track_ids_js};
        var currentTimeMax = 100;
        var plotElement = null;
        
        function findPlotElement() {{
            // Try multiple ways to find the plot
            plotElement = document.getElementById('plotDiv');
            if (!plotElement) {{
                var plotlyDivs = document.getElementsByClassName('plotly-graph-div');
                if (plotlyDivs.length > 0) {{
                    plotElement = plotlyDivs[0];
                    plotElement.id = 'plotDiv';
                }}
            }}
            return plotElement;
        }}
        
        function updateTrackVisibility() {{
            if (!findPlotElement()) {{
                console.log('Plot element not found');
                return;
            }}
            
            var selectedTrack = document.getElementById('trackFilter').value;
            var showTracks = document.getElementById('showTracks').checked;
            
            console.log('Updating track visibility - selected:', selectedTrack, 'show tracks:', showTracks);
            
            var update = {{}};
            var traceCount = plotElement.data ? plotElement.data.length : 0;
            
            var visibilityArray = [];
            
            for (var i = 0; i < traceCount; i++) {{
                var traceName = plotElement.data[i].name || '';
                var isVisible = true; // Default to visible
                
                console.log('Trace', i, ':', traceName);
                
                if (traceName.includes('Track ') || traceName.includes('Start') || traceName.includes('End')) {{
                    // Check if this trace belongs to selected track
                    if (selectedTrack !== 'all') {{
                        isVisible = traceName.includes('Track ' + selectedTrack);
                    }}
                    
                    // Check if tracks are enabled
                    if (!showTracks) {{
                        isVisible = false;
                    }}
                    
                    console.log('Setting trace', i, traceName, 'to', isVisible);
                }}
                
                visibilityArray.push(isVisible);
            }}
            
            console.log('Applying visibility array:', visibilityArray);
            Plotly.restyle(plotElement, 'visible', visibilityArray);
        }}
        
        function updateElementVisibility() {{
            if (!findPlotElement()) {{
                console.log('Plot element not found for element visibility');
                return;
            }}
            
            var showDetections = document.getElementById('showDetections') ? document.getElementById('showDetections').checked : true;
            var showGroundTruth = document.getElementById('showGroundTruth') ? document.getElementById('showGroundTruth').checked : true;
            var showSensors = document.getElementById('showSensors').checked;
            var showBoundary = document.getElementById('showBoundary').checked;
            
            console.log('Element visibility - detections:', showDetections, 'ground truth:', showGroundTruth, 'sensors:', showSensors, 'boundary:', showBoundary);
            
            var traceCount = plotElement.data ? plotElement.data.length : 0;
            var visibilityArray = [];
            
            for (var i = 0; i < traceCount; i++) {{
                var traceName = plotElement.data[i].name || '';
                var isVisible = true; // Start with current visibility or default to true
                
                // Check each element type and hide if needed
                if (traceName === 'Detections') {{
                    isVisible = showDetections;
                    console.log('Detections trace found:', traceName, 'setting to', isVisible);
                }} else if (traceName && traceName.includes('Ground Truth')) {{
                    isVisible = showGroundTruth;
                    console.log('Ground Truth trace found:', traceName, 'setting to', isVisible);
                }} else if (traceName && traceName.includes('Sensor')) {{
                    isVisible = showSensors;
                    console.log('Sensor trace found:', traceName, 'setting to', isVisible);
                }} else if (traceName === 'Field Boundary') {{
                    isVisible = showBoundary;
                    console.log('Field Boundary trace found:', traceName, 'setting to', isVisible);
                }} else {{
                    // For track-related traces, keep them visible (track filter handles them)
                    console.log('Other trace (keeping visible):', traceName);
                }}
                
                visibilityArray.push(isVisible);
            }}
            
            console.log('Applying element visibility array:', visibilityArray);
            Plotly.restyle(plotElement, 'visible', visibilityArray).then(function() {{
                // Don't call updateTrackVisibility here - it would override our changes
                // Instead, apply track filter separately after this completes
                setTimeout(function() {{
                    var selectedTrack = document.getElementById('trackFilter').value;
                    var showTracks = document.getElementById('showTracks').checked;
                    
                    if (selectedTrack !== 'all' || !showTracks) {{
                        applyTrackFilterOnly();
                    }}
                }}, 100);
            }});
        }}
        
        function applyTrackFilterOnly() {{
            if (!findPlotElement()) return;
            
            var selectedTrack = document.getElementById('trackFilter').value;
            var showTracks = document.getElementById('showTracks').checked;
            var traceCount = plotElement.data ? plotElement.data.length : 0;
            
            // Only update track-related traces
            var trackUpdates = {{}};
            
            for (var i = 0; i < traceCount; i++) {{
                var traceName = plotElement.data[i].name || '';
                
                if (traceName.includes('Track ') || traceName.includes('Start') || traceName.includes('End')) {{
                    var isVisible = true;
                    
                    if (selectedTrack !== 'all') {{
                        isVisible = traceName.includes('Track ' + selectedTrack);
                    }}
                    
                    if (!showTracks) {{
                        isVisible = false;
                    }}
                    
                    trackUpdates[i] = isVisible;
                    console.log('Track filter: Setting trace', i, traceName, 'to', isVisible);
                }}
            }}
            
            // Apply only to track traces
            var indices = Object.keys(trackUpdates).map(Number);
            var values = indices.map(i => trackUpdates[i]);
            
            if (indices.length > 0) {{
                Plotly.restyle(plotElement, 'visible', values, indices);
            }}
        }}
        
        function updateTimeFilter() {{
            var timeSlider = document.getElementById('timeRange');
            var timePercent = parseInt(timeSlider.value);
            currentTimeMax = timePercent;
            
            updateTimeDisplay();
            
            // This is a simplified time filter - in a full implementation,
            // you would filter the actual data points based on timestamp
            var info = document.getElementById('timeInfo');
            if (timePercent === 100) {{
                info.textContent = "Showing: Full Duration";
            }} else {{
                info.textContent = "Showing: First " + timePercent + "% of session";
            }}
        }}
        
        function updateTimeDisplay() {{
            var timeSlider = document.getElementById('timeRange');
            var timePercent = parseInt(timeSlider.value);
            var info = document.getElementById('timeInfo');
            
            if (timePercent === 100) {{
                info.textContent = "Showing: Full Duration";
            }} else {{
                info.textContent = "Showing: First " + timePercent + "% of session";
            }}
        }}
        
        function resetFilters() {{
            // Reset all controls
            document.getElementById('trackFilter').value = 'all';
            document.getElementById('showTracks').checked = true;
            if (document.getElementById('showDetections')) document.getElementById('showDetections').checked = true;
            if (document.getElementById('showGroundTruth')) document.getElementById('showGroundTruth').checked = true;
            document.getElementById('showSensors').checked = true;
            document.getElementById('showBoundary').checked = true;
            document.getElementById('timeRange').value = 100;
            
            // Update visibility
            updateElementVisibility();
            updateTimeDisplay();
            
            // Reset zoom
            if (plotElement) {{
                Plotly.relayout(plotElement, {{
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                }});
            }}
        }}
        
        function testFilters() {{
            console.log('Testing filters...');
            if (findPlotElement()) {{
                console.log('Plot found, trace count:', plotElement.data ? plotElement.data.length : 0);
                if (plotElement.data) {{
                    for (var i = 0; i < plotElement.data.length; i++) {{
                        console.log('Trace', i, ':', plotElement.data[i].name);
                    }}
                }}
            }} else {{
                console.log('Plot not found');
            }}
        }}
        '''


def analyze_session(data_dir: str, session_name: str, output_dir: str = "analysis"):
    """
    Convenience function to analyze a tracking session.
    
    Args:
        data_dir: Directory containing CSV files
        session_name: Session name
        output_dir: Output directory for analysis
    """
    analyzer = TrackingAnalyzer(data_dir, session_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    report_file = os.path.join(output_dir, f"{session_name}_report.txt")
    analyzer.export_report(report_file)
    
    # Generate scorecard if ground truth is available
    if analyzer.ground_truth_df is not None:
        scorecard_file = os.path.join(output_dir, f"{session_name}_scorecard.png")
        scorecard = analyzer.plot_tracking_scorecard(save_path=scorecard_file)
        
        # Export scorecard metrics to text
        if scorecard:  # Only if scorecard is not None/empty
            scorecard_text_file = os.path.join(output_dir, f"{session_name}_scorecard.txt")
            with open(scorecard_text_file, 'w') as f:
                f.write("# Tracking Performance Scorecard\n\n")
                for key, value in scorecard.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            print(f"Scorecard metrics saved to {scorecard_text_file}")
            print(f"Scorecard analysis saved to {scorecard_file}")
        else:
            print("No scorecard data available - timestamp mismatch between tracks and ground truth")
    else:
        print("No ground truth data available - skipping scorecard analysis")
    
    # Generate interactive trajectory plot
    try:
        interactive_file = os.path.join(output_dir, f"{session_name}_interactive.html")
        html_path = analyzer.plot_interactive_trajectories(
            save_path=interactive_file,
            include_measurements=True,
            include_ground_truth=True
        )
        if html_path:
            print(f"Interactive trajectory plot saved to {html_path}")
    except Exception as e:
        print(f"Could not generate interactive plot: {e}")
        print("Tip: Install plotly with 'pip install plotly' for interactive plots")
    
    print(f"Analysis completed for session: {session_name}")
    print(f"Results saved in: {output_dir}/")
    
    return analyzer


def compare_sessions(sessions: List[Tuple[str, str]], output_dir: str = "comparison"):
    """
    Compare multiple tracking sessions.
    
    Args:
        sessions: List of (data_dir, session_name) tuples
        output_dir: Output directory for comparison
    """
    analyzers = []
    
    for data_dir, session_name in sessions:
        analyzer = TrackingAnalyzer(data_dir, session_name)
        analyzers.append((session_name, analyzer))
    
    # Create comparison plots
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Compare track counts
    plt.subplot(1, 3, 1)
    session_names = []
    track_counts = []
    
    for name, analyzer in analyzers:
        if analyzer.tracks_df is not None:
            session_names.append(name)
            track_counts.append(analyzer.tracks_df['track_id'].nunique())
    
    plt.bar(session_names, track_counts)
    plt.title('Total Tracks by Session')
    plt.xticks(rotation=45)
    
    # Compare average speeds
    plt.subplot(1, 3, 2)
    avg_speeds = []
    
    for name, analyzer in analyzers:
        if analyzer.tracks_df is not None:
            speeds = np.sqrt(analyzer.tracks_df['vel_x']**2 + analyzer.tracks_df['vel_y']**2)
            avg_speeds.append(speeds.mean())
    
    plt.bar(session_names, avg_speeds)
    plt.title('Average Speed by Session')
    plt.xticks(rotation=45)
    
    # Compare session durations
    plt.subplot(1, 3, 3)
    durations = []
    
    for name, analyzer in analyzers:
        if analyzer.tracks_df is not None:
            duration = analyzer.tracks_df['timestamp'].max() - analyzer.tracks_df['timestamp'].min()
            durations.append(duration)
    
    plt.bar(session_names, durations)
    plt.title('Session Duration')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "session_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Session comparison saved to {output_dir}/session_comparison.png")