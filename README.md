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
conda activate texprompting
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
   - There will be a random Reviewer assigned who needs to review and Merge the Code. 

5. **JIRA Integration**
   - Once your PR is linked in the title/description, JIRA should automatically detect the PR
   - A new Subtask is created for the Review. Assign the same Reviewer as in Github to the Subtask




# Versatile Production System (VPS) Dataset

## Overview

This repository contains process data from the **Versatile Production System (VPS)**, a modular smart factory demonstrator from the SmartFactory OWL environment. The system is used in industrial research for applications such as **machine learning, anomaly detection, process monitoring, and alarm management**.

The VPS simulates an end-to-end production workflow for popcorn processing and packaging using interconnected industrial modules.

---

## System Description

The VPS consists of five main modules:

### Delivery Module
Raw corn is delivered into the system via conveyor belt and transported into a stainless steel funnel. A pressure conveyor moves the material into storage.

### Storage Module
Acts as a buffer for raw material. When sufficient corn is available and downstream capacity allows, material is pneumatically transferred to dosing.

### Dosing Module
A controlled amount of corn is measured using a load cell and dosing screw mechanism. The measured portion is then transferred to the filling module.

### Filling Module
Bottles are processed on a rotary table through multiple stages:

- Cleaning using compressed air  
- Filling with corn  
- Lid placement using a pneumatic gripper  
- Screwing on lids  
- Quality inspection using a camera system  

### Production Module
Corn is heated and expanded into popcorn. The product is collected into cups until a weight threshold is reached. Excess popcorn is redirected to an overflow container.

---

## Dataset Content

The dataset contains time-series process data collected from sensors and actuators across the VPS modules.

Typical data includes:
- Sensor readings (e.g., weight, pressure, levels)
- Actuator states (e.g., motors, valves, conveyors)
- Process states and cycle information
- Control signals from the automation system

> Note: The exact signals and file names may depend on the specific experiment export.

---

## Acknowledgements

This dataset originates from:

- inIT – Institute Industrial IT  
- Ostwestfalen-Lippe University of Applied Sciences  

The VPS is part of the **SmartFactory OWL** research infrastructure.

It has been used in research projects such as **IMPROVE**, funded by the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 678867).

---

## License / Usage

This dataset is publicly available under [these terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Users are responsible for complying with the original data provider’s terms. For commercial use, please consult the original dataset owners.
