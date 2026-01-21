#!/bin/bash
# --------------------------------------------
# Run Streamlit App on macOS automatically
# --------------------------------------------

# Set your project path
PROJECT_PATH="$HOME/Downloads/IG_Prov1"

# Navigate to project folder
cd "$PROJECT_PATH" || exit

# Activate virtual environment if you have one
# Replace 'venv' with your virtual environment folder name
if [ -d "$PROJECT_PATH/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_PATH/venv/bin/activate"
fi

# Run Streamlit app on all network interfaces, port 8501
echo "Starting Streamlit app..."
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Keep terminal open after execution
exec $SHELL