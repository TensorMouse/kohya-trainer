import os
import subprocess
import urllib.request
import accelerate

def clone_or_update_repo(url, update=False, save_directory=None):
    """
    Function to clone or update a repository based on the given URL and update flag.
    
    Parameters:
        url (str): The URL of the repository to clone or update.
        update (bool): Flag indicating whether to update the repository if it already exists.
        save_directory (str): The directory where the repository will be saved. Default is the current working directory.
    
    Returns:
        str: The name of the repository directory.
    """
    # Extract the repository directory name from the URL
    repo_dir = url.split('/')[-1].split('.')[0]

    # Set the save path by joining the save directory and repository directory name
    if save_directory is None:
        save_path = repo_dir
    else:
        save_path = os.path.join(save_directory, repo_dir)
    # Check if the repository directory doesn't exist
    # Clone the repository if it doesn't exist
    if not os.path.exists(save_path):
        subprocess.run(['git', 'clone', url, save_path])
        
    # Check if the repository directory exists and update flag is True
    # Update the repository if it already exists and update flag is True
    elif os.path.exists(save_path) and update:
        subprocess.run(['git', '-C', save_path, 'pull'])
        
    # The repository directory exists and update flag is False
    # Display a message if the repository already exists and update flag is False
    else:
        print(f"{repo_dir} folder already exists")
    
    return save_path

def install_dependencies(install_dir=None, verbose=False, install_xformers=False):
    """
    Function to install dependencies required by the application.

    Parameters:
        install_dir (str): The directory where the dependencies will be installed. Default is None.
        verbose (bool): Flag indicating whether to show verbose output during installation. Default is False.
        install_xformers (bool): Flag indicating whether to install additional xformers dependencies. Default is False.

    Returns:
        None
    """
    print('Installation can take multiple minutes, enable "Verbose" to see progress')
    s = subprocess.getoutput('nvidia-smi')

    if 'T4' in s:
        sed_command = "sed -i 's@cpu@cuda@' library/model_util.py"
        subprocess.run(sed_command, shell=True, check=True)

    pip_install_command = f"pip install {'-q' if not verbose else ''} --upgrade -r requirements.txt"
    subprocess.run(pip_install_command, shell=True, check=True)

    pytorch_install_command = f"pip install {'-q' if not verbose else ''} torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U"
    subprocess.run(pytorch_install_command, shell=True, check=True)

    conda_install_command = "conda install -c conda-forge glib --yes"
    subprocess.run(conda_install_command, shell=True, check=True)
    
    if install_xformers:
        xformers_install_command = f"pip install {'-q' if not verbose else ''} xformers==0.0.19 triton==2.0.0 -U"
        subprocess.run(xformers_install_command, shell=True, check=True)
        
    from accelerate.utils import write_basic_config

    #accelerate_config=os.path.join(install_dir,accelerate_config)
    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)
