# texprompter


## Installation Guide for Conda env:
🛠️ How to Set Up the Project Environment
To ensure everyone on the team has the exact same package versions regardless of whether you are on Windows, Linux, or macOS, we are using conda-lock. This prevents "it works on my machine" bugs.

Please follow these steps to set up your environment:

Step 1: Install conda-lock in your base environment
You only need to do this once on your machine. Open your terminal (or Anaconda Prompt on Windows) and run:

Bash
conda install -c conda-forge conda-lock
Step 2: Navigate to the project folder
Make sure you have pulled the latest code and use your terminal to navigate into the directory containing the conda-lock.yml file.

Bash
cd path/to/the/project/folder
Step 3: Install the locked environment
Run the following command. conda-lock will automatically detect your operating system and install the exact matching dependencies:

Bash
conda-lock install --name texprompting conda-lock.yml
(Note: If you want to use a different name for your local environment, you can change shared_project_env to whatever you prefer).

Step 4: Activate your new environment
Once the installation finishes, activate it just like a normal Conda environment:

Bash
conda activate shared_project_env
