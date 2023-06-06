import os
import subprocess
import urllib.request
import accelerate

def create_dirs(root_dir="/home/studio-lab-user/sagemaker-studiolab-notebooks"):
    """
    Function to create necessary directories for the project.

    Parameters:
        root_dir (str): Root directory where the project directories will be created.
                        Default is "/home/studio-lab-user/sagemaker-studiolab-notebooks".

    Returns:
        dict: A dictionary containing the directory names as keys and their respective paths as values.
    """

    # Define directory paths
    deps_dir = os.path.join(root_dir, "deps")
    repo_dir = os.path.join(root_dir, "kohya-trainer")
    pretrained_dir = os.path.join(root_dir, "pretrained_model")
    vae_dir = os.path.join(root_dir, "vae")
    trainer_config = os.path.join(repo_dir, "trainer_config.ini")

    dreambooth_training_dir = os.path.join(root_dir, "dreambooth")
    dreambooth_config_dir = os.path.join(dreambooth_training_dir, "config")
    dreambooth_output_dir = os.path.join(dreambooth_training_dir, "output")
    dreambooth_sample_dir = os.path.join(dreambooth_output_dir, "sample")
    dreambooth_logging_dir = os.path.join(dreambooth_training_dir, "logs")

    inference_dir = os.path.join(root_dir, "txt2img")

    train_data_dir = os.path.join(root_dir, "train_data")
    reg_data_dir = os.path.join(root_dir, "reg_data")

    accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
    tools_dir = os.path.join(repo_dir, "tools")
    finetune_dir = os.path.join(repo_dir, "finetune")

    # Create directories
    for dir in [
        deps_dir,
        dreambooth_training_dir,
        dreambooth_config_dir,
        dreambooth_output_dir,
        pretrained_dir,
        dreambooth_sample_dir,
        inference_dir,
        vae_dir,
        train_data_dir,
        reg_data_dir,
    ]:
        os.makedirs(dir, exist_ok=True)

    # Create and return directory dictionary
    dirs = {
        "deps_dir": deps_dir,
        "repo_dir": repo_dir,
        "pretrained_dir": pretrained_dir,
        "vae_dir": vae_dir,
        "trainer_config": trainer_config,
        "dreambooth_training_dir": dreambooth_training_dir,
        "dreambooth_config_dir": dreambooth_config_dir,
        "dreambooth_output_dir": dreambooth_output_dir,
        "dreambooth_sample_dir": dreambooth_sample_dir,
        "dreambooth_logging_dir": dreambooth_logging_dir,
        "inference_dir": inference_dir,
        "train_data_dir": train_data_dir,
        "reg_data_dir": reg_data_dir,
        "accelerate_config": accelerate_config,
        "tools_dir": tools_dir,
        "finetune_dir": finetune_dir,
    }

    return dirs


def clone_or_update_repo(url, update=False, save_directory=None, branch=None):
    """
    Function to clone or update a repository based on the given URL, update flag, and branch.
    
    Parameters:
        url (str): The URL of the repository to clone or update.
        update (bool): Flag indicating whether to update the repository if it already exists.
        save_directory (str): The directory where the repository will be saved. Default is the current working directory.
        branch (str): The branch to clone or update. Default is None (which uses the default branch).
    
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
        if branch is None:
            subprocess.run(['git', 'clone', url, save_path])
        else:
            subprocess.run(['git', 'clone', '-b', branch, url, save_path])
        
    # Check if the repository directory exists and update flag is True
    # Update the repository if it already exists and update flag is True
    elif os.path.exists(save_path) and update:
        if branch is None:
            subprocess.run(['git', '-C', save_path, 'pull'])
        else:
            subprocess.run(['git', '-C', save_path, 'pull', '-b', branch])
        
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

def download_model(url, save_directory):
    """
    Function to download a model file from a given URL to a specified directory.
    Parameters:
        url (str): The URL of the model file to download.
        save_directory (str): The directory where the model file will be saved.
    """
    # Extract the file name from the URL
    file_name = url.split('/')[-1]
    
    # Create the save path by joining the save directory and file name
    save_path = os.path.join(save_directory, file_name)

    # Check if the model file doesn't exist
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        # Download the model file from the URL
        urllib.request.urlretrieve(url, save_path)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")
        
def preProcessingParams():
    """
    Retrieves the pre-processing parameters.

    Returns:
        tuple: A tuple containing the batch size, supported types, and background colors.

    Example:
        batch_size, supported_types, background_colors = preProcessingParams()
    """
    batch_size = 32
    supported_types = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".caption",
        ".npz",
        ".txt",
        ".json",
    ]

    background_colors = [
        (255, 255, 255),
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    
    return batch_size, supported_types, background_colors

def clean_directory(directory, supported_types):
    """
    Clean the specified directory by removing files with unsupported types.

    This function iterates through all items (files and subdirectories) in the given directory and removes any files
    with extensions that are not included in the list of supported types.

    Parameters:
        directory (str): The directory to clean.
        supported_types (list): A list of supported file extensions (e.g., ['.png', '.jpg', '.txt']).

    Returns:
        None
    """

    for item in os.listdir(directory):
        file_path = os.path.join(directory, item)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(item)[1]
            
            if file_ext not in supported_types:
                print(f"Deleting file {item} from {directory}")
                os.remove(file_path)

        elif os.path.isdir(file_path):
            # Recursive call to clean subdirectories
            clean_directory(file_path, supported_types)

def process_image(image_path, background_colors):
    """
    Process an image by applying background color and converting it to a supported format.

    This function opens the image at the specified path, applies a background color based on the image mode,
    and converts the image to a supported format if necessary.

    Parameters:
        image_path (str): The path to the image file.
        background_colors (list): A list of background colors in RGB format.

    Returns:
        None
    """

    img = Image.open(image_path)
    img_dir, image_name = os.path.split(image_path)

    if img.mode in ("RGBA", "LA"):
        if random_color:
            background_color = random.choice(background_colors)
        else:
            background_color = (255, 255, 255)

        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[-1])

        if image_name.endswith(".webp"):
            bg = bg.convert("RGB")
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            bg.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            bg.save(image_path, "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp"):
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            img.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            img.save(image_path, "PNG")

def find_images(directory):
    """
    Find all image files (PNG and WebP) within a directory and its subdirectories.

    This function recursively searches for image files with the extensions ".png" and ".webp" within the specified directory
    and its subdirectories. It returns a list of the absolute paths to these image files.

    Parameters:
        directory (str): The directory to search for image files.

    Returns:
        list: A list of absolute paths to the found image files.
    """

    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".webp"):
                images.append(os.path.join(root, file))
    return images

def convertImages(images, batch_size, num_batches):
    """
    Convert a batch of images using multithreading.

    This function takes a list of image file paths, divides them into batches based on the specified batch size,
    and converts each image in parallel using multithreading.

    Parameters:
        images (list): A list of image file paths to be converted.
        batch_size (int): The number of images to process in each batch.
        num_batches (int): The total number of batches to process.

    Returns:
        None
    """

    if convert:
        # Create a ThreadPoolExecutor to parallelize image processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in tqdm(range(num_batches)):
                start = i * batch_size
                end = start + batch_size
                batch = images[start:end]
                executor.map(process_image, batch)

    print("All images have been converted")
