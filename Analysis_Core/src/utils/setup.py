"""
Setup utilities for the blood cell classification system.

This module includes functions for installation, environment setup, 
and other initialization processes.
"""

import os
import gc
from IPython.utils import capture
import psutil

def install(pkg):
    """
    Install a Python package if not already installed.
    
    Args:
        pkg (str): Package name to install
    """
    from IPython.utils import capture
    try:
        __import__(pkg)
        print(f"{pkg} is already installed.")
    except ImportError:
        print(f"Installing {pkg}...")
        with capture.capture_output() as captured:
            # This would be a !pip install {pkg} in Jupyter/Colab
            # Using os.system for compatibility in regular Python
            os.system(f"pip install {pkg}")

def print_memory_usage():
    """Prints the memory usage of the current process in MB"""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.2f} MB")
    print(f"Shared Memory Size: {memory_info.shared / 1024**2:.2f} MB")

def print_memory_usage_msg(message):
    """Print memory usage with a custom message"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message} - Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def setup_colab_environment(dataset_identifier):
    """
    Sets up Kaggle integration and downloads the dataset in Google Colab.
    
    Args:
        dataset_identifier (str): Kaggle dataset identifier
    """
    try:
        import os
        from google.colab import files
        os.system("pip install -q kaggle")
        
        # Define Kaggle directory and JSON path
        kaggle_dir = os.path.expanduser("~/.kaggle")
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        
        # Ensure Kaggle directory exists and has proper permissions
        os.makedirs(kaggle_dir, exist_ok=True)
        if os.path.exists(kaggle_json_path):
            print("kaggle.json found. Ensuring proper configuration.")
            os.system("chmod 600 ~/.kaggle/kaggle.json")
        else:
            print("kaggle.json not found. Please upload the Kaggle API key (kaggle.json):")
            uploaded = files.upload()
            
            if 'kaggle.json' not in uploaded:
                raise ValueError("Kaggle API key file (kaggle.json) is required.")
            
            # Move uploaded file to the Kaggle directory
            os.system("mv kaggle.json ~/.kaggle/")
            os.system("chmod 600 ~/.kaggle/kaggle.json")
        
        print(f"Downloading dataset: {dataset_identifier}")
        
        with capture.capture_output() as captured:
            # Download the specified dataset
            os.system(f"kaggle datasets download -d {dataset_identifier}")
            
            # Unzip the dataset
            zip_file = dataset_identifier.split('/')[-1] + ".zip"
            os.system(f"unzip -o {zip_file}")
        
        print("Dataset setup complete!")
    except ImportError:
        print("Google Colab not detected. Please download the dataset manually.")

def push_notebook_to_github(notebook_name, commit_message=None, branch='colab_branch'):
    """
    Push a Jupyter notebook to GitHub from Google Colab.
    
    Args:
        notebook_name (str): Name of the notebook file
        commit_message (str, optional): Git commit message
        branch (str, optional): Git branch name
    """
    try:
        from google.colab import userdata, files
        
        try:
            github_token = userdata.get('GITHUB_TOKEN')
            repo_path = userdata.get('GITHUB_REPO')
            notebook_name = userdata.get('NOTEBOOK_NAME')
        except Exception as secret_error:
            print("❌ Error retrieving GitHub secrets:")
            print(f"   {secret_error}")
            print("Please ensure you've set up both 'GITHUB_TOKEN' and 'GITHUB_REPO' secrets in Colab")
            return
        
        # Validate credentials
        if not github_token or not repo_path:
            print("❌ Missing GitHub token or repository path.")
            print("Please add 'GITHUB_TOKEN' and 'GITHUB_REPO' in Colab Secrets")
            return
        
        # Use os.system for git commands
        os.system(f"git config --global user.email \"$(git config user.email || echo 'colab@example.com')\"")
        os.system(f"git config --global user.name \"$(git config user.name || echo 'Colab User')\"")
        
        # Set up repository URL with token
        repo_url = f"https://{github_token}@github.com/{repo_path}.git"
        
        # Clone or update the repository
        os.system(f"git clone {repo_url} colab_repo || (cd colab_repo && git pull)")
        
        # Copy the notebook to the repository
        os.system(f"cp \"../{notebook_name}\" colab_repo/")
        
        # Stage the notebook
        os.chdir("colab_repo")
        os.system(f"git add \"{notebook_name}\"")
        
        # Create commit message if not provided
        if commit_message is None:
            from datetime import datetime
            commit_message = f"Update notebook from Colab - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Commit changes
        os.system(f"git commit -m \"{commit_message}\"")
        
        # Push to specified branch
        os.system(f"git push origin {branch}")
        
        print(f"✅ Notebook {notebook_name} successfully pushed to {repo_path} on {branch}")
    
    except ImportError:
        print("Google Colab not detected. This function only works in Colab environment.")
    except Exception as e:
        print(f"❌ Error pushing notebook to GitHub: {e}")

def print_config(config):
    """
    Print the configuration settings.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\n=== CONFIGURATION ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=====================\n") 