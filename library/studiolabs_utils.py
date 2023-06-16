import os
import subprocess
import configparser
import concurrent.futures
import urllib.request
import time

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
    default_configs_dir = os.path.join(repo_dir, "default_configs")
    pretrained_dir = os.path.join(root_dir, "pretrained_model")
    vae_dir = os.path.join(root_dir, "vae")
    default_config = os.path.join(default_configs_dir, "default_config.ini")

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
        root_dir,
        deps_dir,
        default_configs_dir,
        dreambooth_training_dir,
        dreambooth_config_dir,
        dreambooth_output_dir,
        pretrained_dir,
        dreambooth_sample_dir,
        inference_dir,
        vae_dir,
        train_data_dir,
        accelerate_config,
        reg_data_dir,
    ]:
        os.makedirs(dir, exist_ok=True)

    # Create and return directory dictionary
    dirs = {
        "root_dir" : root_dir,
        "deps_dir": deps_dir,
        "repo_dir": repo_dir,
        "pretrained_dir": pretrained_dir,
        "default_configs_dir" : default_configs_dir,
        "vae_dir": vae_dir,
        "default_config": default_config,
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


def install_dependencies(dirs, verbose=False, install_xformers=False):
    """
    Function to install dependencies required by the application.

    Parameters:
        dirs (dict): The dictionary with path variables.
        verbose (bool): Flag indicating whether to show verbose output during installation. Default is False.
        install_xformers (bool): Flag indicating whether to install additional xformers dependencies. Default is False.

    Returns:
        None
    """
    
    print('Installation can take multiple minutes, enable "Verbose" to see progress')
    s = subprocess.getoutput('nvidia-smi')
    
    util_file = os.path.join(dirs['repo_dir'], "library/model_util.py")
    req_file = os.path.join(dirs['repo_dir'], "requirements.txt")
    if 'T4' in s:
        sed_command = f"sed -i 's@cpu@cuda@' {util_file}"
        subprocess.run(sed_command, shell=True, check=True)

    pip_install_command = f"pip install {'-q' if not verbose else ''} --upgrade -r {req_file}"
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
    if not os.path.exists(dirs['accelerate_config']):
        write_basic_config(save_location=dirs['accelerate_config'])

def download_model(url, save_directory):
    """
    Function to download a model file from a given URL to a specified directory.
    Parameters:
        url (str): The URL of the model file to download.
        save_directory (str): The directory where the model file will be saved.
    """
    try:
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
    except Exception as e:
        print(f"Failed to download the model from {url}. Error: {str(e)}")
        
def get_config_from_folder(folder_path, config_name = 'config.ini'):
    config = configparser.ConfigParser()
    config_file = os.path.join(folder_path, config_name)
    if os.path.isfile(config_file):
        print('folder config exists: ', config_file)
        config.read(config_file)
        return config
    else:
        print('folder config does NOT exist: ', config_file)
        return None
    
        
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
        ".ini",
        ".toml"
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

def convertImages(images, convert, batch_size, num_batches):
    """
    Convert a batch of images using multithreading.

    This function takes a list of image file paths, divides them into batches based on the specified batch size,
    and converts each image in parallel using multithreading.

    Parameters:
        images (list): A list of image file paths to be converted.
        convert (bool): Indicates whether image conversion should be performed.
        batch_size (int): The number of images to process in each batch.
        num_batches (int): The total number of batches to process.

    Returns:
        None
    """
    from tqdm import tqdm
    if convert:
        # Create a ThreadPoolExecutor to parallelize image processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in tqdm(range(num_batches)):
                start = i * batch_size
                end = start + batch_size
                batch = images[start:end]
                executor.map(process_image, batch)

    print("All images have been converted")
    
def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)
        
def process_tags(filename, custom_tag, append, prefix_tag, remove_tag, keywords):
    """
    Processes tags in a file based on specified options.

    Args:
        filename (str): The name of the file to process.
        custom_tag (str): The custom tag to modify existing tags or append/prefix to the tag list.
        append (bool): Indicates whether to append the custom tag to the tag list.
        prefix_tag (bool): Indicates whether to prefix the custom tag to the tag list.
        remove_tag (bool): Indicates whether to remove occurrences of the custom tag from the tag list.
        keywords (list): A list of keywords to search for in existing tags.

    Returns:
        None
    """
    contents = read_file(filename)
    tags = [tag.strip() for tag in contents.split(',')]
    custom_tags = [tag.strip() for tag in custom_tag.split(',')]
    for custom_tag in custom_tags:
        custom_tag = custom_tag.replace("_", " ")
        if remove_tag:
            while custom_tag in tags:
                tags.remove(custom_tag)
        else:
            for i in range(len(tags)):
                for keyword in keywords:
                    if keyword in tags[i]:
                        tags[i] = tags[i].replace(keyword, custom_tag)
            if append:
                tags.append(custom_tag)
            if prefix_tag:
                tags.insert(0, custom_tag)

    contents = ', '.join(tags)
    write_file(filename, contents)


def process_directory(image_dir, tag, append, prefix_tag, remove_tag, recursive, keyword, extension):
    """
    Processes tags in files within a directory based on specified options.

    Args:
        image_dir (str): The path to the directory containing the files.
        tag (str): The custom tag to modify existing tags or append/prefix to the tag list.
        append (bool): Indicates whether to append the custom tag to the tag list.
        prefix_tag (bool): Indicates whether to prefix the custom tag to the tag list.
        remove_tag (bool): Indicates whether to remove occurrences of the custom tag from the tag list.
        recursive (bool): Indicates whether to process files in subdirectories recursively.
        keyword (str): The keyword to search for in existing tags.

    Returns:
        None
    """
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        
        if os.path.isdir(file_path) and recursive:
            process_directory(file_path, tag, append, prefix_tag, remove_tag, recursive, keyword)
        elif filename.endswith(extension):
            process_tags(file_path, tag, append, prefix_tag, remove_tag, keyword)
            

def custom_caption_tag(config, train_image_folder):
    """
    Performs custom caption/tag operations based on the configurations specified in the provided config object.

    Args:
        config (configparser.ConfigParser): The config object containing the parameter values.
        
    Config Parameters:
        extension: The file extension for caption/tag files
        custom_tag:  The custom tag to be added, modified, or removed
        keywords: Keywords used to identify tags for modification
        sub_folder: The subfolder within the image directory to process
        append: Indicates whether to append the custom tag
        prefix_tag: Indicates whether to prefix the custom tag
        remove_tag: Indicates whether to remove the custom tag
        recursive: Indicates whether to process subdirectories recursively

    Returns:
        None
    """

    # Retrieve the parameter values from the config object
    extension = config.get('CustomCaptionTag', 'extension')
    custom_tag = config.get('CustomCaptionTag', 'custom_tag')
    keywords = config.get('CustomCaptionTag', 'keywords').split(',')
    sub_folder = config.get('CustomCaptionTag', 'sub_folder')
    append = config.getboolean('CustomCaptionTag', 'append')
    prefix_tag = config.getboolean('CustomCaptionTag', 'prefix_tag')
    remove_tag = config.getboolean('CustomCaptionTag', 'remove_tag')
    recursive = config.getboolean('CustomCaptionTag', 'recursive')

    # Determine the image directory based on the sub_folder parameter
    if sub_folder == "None":
        image_dir = train_image_folder
    elif sub_folder == "--all":
        image_dir = train_image_folder
        recursive = True
    else:
        image_dir = os.path.join(train_image_folder, sub_folder)
        os.makedirs(image_dir, exist_ok=True)

    # Create missing caption/tag files if necessary
    if not any([filename.endswith(extension) for filename in os.listdir(image_dir)]):
        for filename in os.listdir(image_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                open(os.path.join(image_dir, filename.split(".")[0] + extension), "w").close()

    # Process custom tags/directories
    if custom_tag:
        process_directory(image_dir, custom_tag, append, prefix_tag, remove_tag, recursive, keywords, extension)


def run_captioning_process(config, train_image_folder, finetune_dir):
    """
    Runs the captioning process based on the configurations specified in the provided config object.

    Args:
        config (configparser.ConfigParser): The config object containing the parameter values.
        train_image_folder (str): The path to the training image folder.
        finetune_dir (str): The path to the directory containing the captioning scripts.

    Returns:
        None
    """

    # Retrieve the captioning type from the config object
    captioning_type = config.get('Captioning', 'captioning')

    # Check the captioning type and set the ArgsConfig accordingly
    
    # Config Parameters:
    #     _train_data_dir: The directory containing the training images
    #     batch_size: The batch size for the captioning process
    #     beam_search: Indicates whether to use beam search during caption generation
    #     min_length: The minimum length of generated captions
    #     max_length: The maximum length of generated captions
    #     debug: Indicates whether to enable verbose logging
    #     caption_extension: The file extension for caption files
    #     max_data_loader_n_workers: The maximum number of data loader workers
    #     recursive: Indicates whether to process subdirectories recursively

    if captioning_type == "BLIP":
        ArgsConfig = {
            "_train_data_dir": train_image_folder,
            "batch_size": config.getint('Captioning', 'batch_size'),
            "beam_search": config.getboolean('Captioning', 'beam_search'),
            "min_length": config.getint('Captioning', 'min_length'),
            "max_length": config.getint('Captioning', 'max_length'),
            "debug": config.getboolean('Captioning', 'verbose_logging'),
            "caption_extension": config.get('Captioning', 'caption_extension'),
            "max_data_loader_n_workers": config.getint('Captioning', 'max_data_loader_n_workers'),
            "recursive": config.getboolean('Captioning', 'recursive')
        }
        
    # Config Parameters:
    #     _train_data_dir: The directory containing the training images
    #     batch_size: The batch size for the captioning process
    #     beam_search: Indicates whether to use beam search during caption generation
    #     min_length: The minimum length of generated captions
    #     max_length: The maximum length of generated captions
    #     debug: Indicates whether to enable verbose logging
    #     caption_extension: The file extension for caption files
    #     max_data_loader_n_workers: The maximum number of data loader workers
    #     recursive: Indicates whether to process subdirectories recursively

    elif captioning_type == "Waifu":
        ArgsConfig = {
            "_train_data_dir": train_image_folder,
            "batch_size": config.getint('Captioning', 'batch_size'),
            "repo_id": config.get('Captioning', 'model'),
            "recursive": config.getboolean('Captioning', 'recursive'),
            "remove_underscore": True,
            "general_threshold": config.getfloat('Captioning', 'general_threshold'),
            "character_threshold": config.getfloat('Captioning', 'character_threshold'),
            "caption_extension": config.get('Captioning', 'caption_extension'),
            "max_data_loader_n_workers": config.getint('Captioning', 'max_data_loader_n_workers'),
            "debug": config.getboolean('Captioning', 'verbose_logging'),
            "undesired_tags": config.get('Captioning', 'undesired_tags')
        }
    else:
        print("No captioning option selected. Skipping captioning process.")
        return
    
    # Build the command-line arguments based on ArgsConfig
    args = ""
    for k, v in ArgsConfig.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    # Define the paths to the captioning scripts
    BLIP_captions = os.path.join(finetune_dir, 'make_captions.py')
    WAIFU_captions = os.path.join(finetune_dir, 'tag_images_by_wd14_tagger.py')

    # Define the final command based on the captioning type
    print(args)
    final_args = f"python {BLIP_captions} {args}" if captioning_type == "BLIP" else f"python {WAIFU_captions} {args}"
    print(final_args)
    time.sleep(10)

    # Run the captioning process
    subprocess.run(final_args, shell=True, check=True)
    
def preprocess_folder(folder, dirs, default_processing_config):
    """
    Process the specified folder based on the provided configurations.

    Args:
        folder (str): The folder to be processed.
        dirs (dict): A dictionary containing directory paths.
        default_processing_config (configparser.ConfigParser): The default processing configuration.

    Returns:
        None
    """

    # Get folder-specific configuration or use the default if not available
    folder_config = get_config_from_folder(folder, config_name = 'config.ini')
    if folder_config is None:
        folder_config = default_processing_config
        
    # Extract parameters from the folder configuration
    convert = folder_config.get('ImagePreprocessing', 'convert')
    random_color = folder_config.get('ImagePreprocessing', 'random_color')
    recursive = folder_config.get('ImagePreprocessing', 'recursive')

    # Preprocessing parameters
    batch_size, supported_types, background_colors = preProcessingParams()

    # Clean the directory before processing
    clean_directory(folder, supported_types)

    # Find images in the folder
    images = find_images(folder)

    # Calculate the number of batches
    num_batches = len(images) // batch_size + 1

    # Convert the images
    convertImages(images, convert, batch_size, num_batches)

    # Run captioning process
    run_captioning_process(folder_config, folder, dirs['finetune_dir'])

    # Apply custom caption tag
    custom_caption_tag(folder_config, folder)
    

def get_config_file_paths(folder, dirs):
    """
    Retrieves the file paths for sample_prompt.txt, config_file.toml, and dataset_config.toml.

    Args:
        folder (str): The folder path to search for the files.
        dirs (dict): A dictionary containing directory paths, including 'default_configs_dir'.

    Returns:
        tuple: A tuple containing the file paths for sample_prompt.txt, config_file.toml, and dataset_config.toml.
    """

    sample_prompt = None
    config_file = None
    if "sample_prompt.txt" in os.listdir(folder):
        sample_prompt = os.path.join(folder, "sample_prompt.txt")
    else:
        sample_prompt = os.path.join(dirs['default_configs_dir'], "sample_prompt.txt")

    if "config.ini" in os.listdir(folder):
        config_file = os.path.join(folder, "config.ini")
    else:
        config_file = None

    return sample_prompt, config_file


def update_model_paths(config, dirs):
    """
    Updates the model paths in the configuration dictionary.

    Args:
        config (dict): The configuration dictionary to update.
        dirs (dict): A dictionary containing directory paths, including 'pretrained_dir' and 'vae_dir'.

    Returns:
        dict: The updated configuration dictionary.

    Raises:
        ValueError: If no model files are found in the specified pretrained directory.
        ValueError: If no VAE model files are found in the specified directory.
    """

    model_files = os.listdir(dirs['pretrained_dir'])
    if model_files:
        pretrained_model_path = os.path.join(dirs['pretrained_dir'], model_files[0])
        config['model_arguments']['pretrained_model_name_or_path'] = pretrained_model_path
    else:
        raise ValueError("No model files found in the specified directory.")

    vae_files = os.listdir(dirs['vae_dir'])
    if vae_files:
        pretrained_vae_path = os.path.join(dirs['vae_dir'], vae_files[0])
        config['model_arguments']['vae'] = pretrained_vae_path
    else:
        raise ValueError("No VAE model files found in the specified directory.")

    return config

def get_config_dict_from_ini(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Create an empty dictionary
    token_dictionary = {}

    # Process each line of the file
    for line in content:
        # Extract the key-value pairs
        key, value = line.strip().split(' = ')
        # Add the key-value pair to the dictionary
        token_dictionary[key] = value

    return token_dictionary


    
def get_train_args(config):
    args = ""
    for k, v in config.items():
        print(k,v)
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args

def fetch_image_locations(directory):
    # Fetch image files from the directory
    image_locations = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    return image_locations

def main(directory,grid_save_dir):
    top_space = 40
    left_space = 400
    text_offset = 4
    padding = 20  # Padding between images
    text_width_limit = 750 