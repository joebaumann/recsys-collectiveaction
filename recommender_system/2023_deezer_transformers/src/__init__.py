import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print("script_dir", script_dir)
# Go up two levels to the 'recommender_system' directory
parent_dir = os.path.dirname(os.path.dirname(script_dir))
print("parent_dir", parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
