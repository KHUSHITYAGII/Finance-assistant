#!/usr/bin/env python3
"""
Setup script for the Finance Assistant project.
This script will check and install all required dependencies.
"""

import subprocess
import sys
import platform
import os

def check_python_version():
    """Check if Python version is at least 3.8"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"You are using Python {current_version[0]}.{current_version[1]}.{current_version[2]}")
        sys.exit(1)
    
    print(f"✓ Python {current_version[0]}.{current_version[1]}.{current_version[2]} detected.")

def install_requirements():
    """Install dependencies from requirements.txt"""
    print("\nInstalling required packages...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found in the current directory.")
        sys.exit(1)
    
    # Install packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All required packages installed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install required packages.")
        sys.exit(1)

def setup_environment_variables():
    """Setup environment variables needed for the project"""
    print("\nSetting up environment variables...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# API Keys\n")
            f.write("ALPHA_VANTAGE_API_KEY=demo\n")
            f.write("OPENAI_API_KEY=\n")
            f.write("\n# Service URLs\n")
            f.write("ORCHESTRATOR_URL=http://localhost:8000\n")
            f.write("API_AGENT_URL=http://localhost:8001\n")
            f.write("SCRAPING_AGENT_URL=http://localhost:8002\n")
            f.write("RETRIEVER_AGENT_URL=http://localhost:8003\n")
            f.write("ANALYSIS_AGENT_URL=http://localhost:8004\n")
            f.write("LANGUAGE_AGENT_URL=http://localhost:8005\n")
            f.write("VOICE_AGENT_URL=http://localhost:8006\n")
        print("✓ Created .env file with default configuration.")
        print("  Please edit .env file to add your API keys.")
    else:
        print("✓ .env file already exists.")

def create_directories():
    """Create necessary directories if they don't exist"""
    print("\nCreating necessary directories...")
    
    directories = [
        "data",
        "data/embeddings",
        "data/filings",
        "data/raw",
        "data/processed"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory} directory.")
        else:
            print(f"✓ {directory} directory already exists.")

def check_torch_installation():
    """Verify PyTorch installation"""
    print("\nVerifying PyTorch installation...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is installed.")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. Detected {torch.cuda.device_count()} CUDA device(s).")
            print(f"  Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("  Note: CUDA is not available. PyTorch will run on CPU mode.")
    except ImportError:
        print("  Warning: Failed to import PyTorch. Voice processing might not work correctly.")

def main():
    """Main setup function"""
    print("="*60)
    print("Finance Assistant Setup")
    print("="*60)
    
    check_python_version()
    install_requirements()
    setup_environment_variables()
    create_directories()
    check_torch_installation()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("You can now run the Finance Assistant by executing:")
    print("  1. Start the services: python main.py")
    print("  2. Launch the Streamlit app: streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    main()