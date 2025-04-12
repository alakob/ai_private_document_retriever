#!/bin/bash
set -e

# Create the visualizations directory if it doesn't exist
mkdir -p /app/visualizations

# If a command is provided, execute it
if [ "$1" = "chat" ]; then
  echo "Starting chat interface on port 7861..."
  echo "Access the chat interface at http://localhost:7861"
  exec python main.py chat
elif [ "$1" = "process" ]; then
  echo "Processing documents..."
  exec python main.py process --dir documents "$@"
elif [ "$1" = "visualize" ]; then
  echo "Generating visualization..."
  # Ensure output goes to mounted volume
  OUTPUT_FILE="/app/visualizations/vector_visualization.html"
  if [[ "$*" == *"--output"* ]]; then
    # User specified an output file, use it as is
    exec python main.py visualize "$@"
  else
    # No output specified, use the visualizations directory
    exec python main.py visualize --output "$OUTPUT_FILE" "${@:2}"
  fi
elif [ "$1" = "search" ]; then
  echo "Searching documents..."
  exec python main.py search "${@:2}"
elif [ "$1" = "api" ]; then
  echo "Starting API server on port 8000..."
  echo "Access the API at http://localhost:8000"
  exec python main.py api --host 0.0.0.0 --port 8000
else
  # Default to starting chat interface
  echo "No specific command provided, starting chat interface by default..."
  echo "Access the chat interface at http://localhost:7861"
  exec python main.py chat
fi
