

import os
import subprocess
import sys


def create_venv(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
        print("Virtual environment created.")


def install_requirements(venv_path):
    """Install packages from requirements.txt into the virtual environment."""
    print("Installing dependencies...")
    # Use os.path.join for cross-platform compatibility
    pip_path = os.path.join(venv_path, 'bin' if os.name == 'posix' else 'Scripts', 'pip')
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
    print("Dependencies installed.")


def run_script_in_venv(venv_path, script_name):
    """Run a Python script using the virtual environment's interpreter."""
    # Determine the correct executable based on the OS
    python_executable = os.path.join(venv_path, 'bin' if os.name == 'posix' else 'Scripts', 'python.exe')
    script_path = os.path.join(os.path.abspath('.'), script_name)
    # Ensure the python executable exists
    if not os.path.exists(python_executable):
        print(f"Expected Python executable not found at {python_executable}")
        return
    # Attempt to run the script
    try:
        subprocess.check_call([python_executable, script_path])
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the script {script_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def run_file_with_dependencies(filename):
    # Change the current working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    venv_path = "venv"
    # Setup and run everything from here
    create_venv(venv_path)
    install_requirements(venv_path)
    run_script_in_venv(venv_path, filename)
    while True:
        # Ask user if they want to run again
        user_input = input("Run again? (Y/N): ").strip().upper()
        if user_input == 'N':
            print("Exiting...")
            break
        if user_input == 'Y':
            print("\nRestarting the process...\n")
            run_script_in_venv(venv_path, filename)