#!/usr/bin/env python3
"""
Standalone script for analyzing tracking data from CSV files.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from astraPro.visualizer.analysis import analyze_session, compare_sessions


def main():
    parser = argparse.ArgumentParser(description='Analyze tracking data from CSV files')
    parser.add_argument('data_dir', help='Directory containing CSV files')
    parser.add_argument('session_name', help='Session name (prefix of CSV files)')
    parser.add_argument('--output', '-o', default='analysis', 
                       help='Output directory for analysis results')
    parser.add_argument('--compare', nargs='+', metavar=('DIR:SESSION'), 
                       help='Compare with other sessions (format: data_dir:session_name)')
    
    args = parser.parse_args()
    
    print(f"Analyzing tracking session: {args.session_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    
    # Analyze main session
    analyzer = analyze_session(args.data_dir, args.session_name, args.output)
    
    # Compare sessions if requested
    if args.compare:
        print(f"\nComparing with {len(args.compare)} other sessions...")
        
        sessions = [(args.data_dir, args.session_name)]
        
        for comp_spec in args.compare:
            if ':' in comp_spec:
                comp_dir, comp_session = comp_spec.split(':', 1)
            else:
                comp_dir = args.data_dir
                comp_session = comp_spec
            
            sessions.append((comp_dir, comp_session))
        
        compare_sessions(sessions, os.path.join(args.output, "comparison"))
    
    print(f"\nAnalysis complete! Results saved in: {args.output}/")


if __name__ == "__main__":
    main()