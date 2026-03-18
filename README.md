# texprompter


## Installation Guide for Conda env:
🛠️ How to Set Up the Project Environment
To ensure everyone on the team has the exact same package versions regardless of whether you are on Windows, Linux, or macOS, we are using conda-lock. This prevents "it works on my machine" bugs.

Please follow these steps to set up your environment:

Step 1: Install conda-lock in your base environment
You only need to do this once on your machine. Open your terminal (or Anaconda Prompt on Windows) and run:

``` Bash
conda install -c conda-forge conda-lock
```
Step 2: Navigate to the project folder
Make sure you have pulled the latest code and use your terminal to navigate into the directory containing the conda-lock.yml file.

``` Bash
cd path/to/the/project/folder
```

Step 3: Install the locked environment
Run the following command. conda-lock will automatically detect your operating system and install the exact matching dependencies:

``` Bash
conda-lock install --name texprompting conda-lock.yml
``` 
(Note: If you want to use a different name for your local environment, you can change shared_project_env to whatever you prefer).

Step 4: Activate your new environment
Once the installation finishes, activate it just like a normal Conda environment:

``` Bash
conda activate shared_project_env
``` 



## Contributing Guide

### Workflow for Contributors

Follow these steps when working on the project:

1. **Clone or Pull the Repository**
   - If you're new: `git clone <repo-url>` and follow the Installation Guide above
   - If you already have it: `git pull origin main` to get the latest changes

2. **Create a New Branch for Your Ticket**
   - Pull the latest main: `git pull origin main`
   - Create a new branch named after your ticket (e.g., `git checkout -b TICKET-ID-short-description`)
   - Use the format: `TICKET-ID-short-description` (kebab-case)

3. **Work on Your Changes**
   - Make your commits with clear, descriptive messages
   - Push your branch regularly: `git push origin TICKET-ID-short-description`

4. **Create a Pull Request**
   - Go to the repository on GitHub/GitLab and create a PR from your branch to `main`
   - Reference the JIRA ticket in the PR title or description (e.g., "MALOCHE-8: Add README")
   - Request code review from teammates
   - Ensure CI/tests pass before merging

5. **JIRA Integration** *(automatic)*
   - Once your PR is linked in the title/description, JIRA should automatically detect the PR
   - Upon merge, the ticket status may update automatically (depends on your JIRA webhook setup)
   - Verify your ticket is updated to the correct status (Done/Closed)


---
Test Change


