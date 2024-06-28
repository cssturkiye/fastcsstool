# Fast CSS Tool
![Logo](https://github.com/cssturkiye/fastcsstool/assets/53001810/20a33f92-3d37-48eb-8c2d-e6910fc6a92c)

## Overview
👋🏻 Welcome to the Field of Computational Social Science!

Fast CSS Tool is here to simplify your data analysis journey. This intuitive application is designed to assist social scientists in analyzing digital datasets, including social media data, with ease.

Even if you have little to no coding experience, Fast CSS Tool makes it easy to preprocess, filter, and classify your data. The tool automates many complex backend processes, leveraging machine learning algorithms to streamline your workflow.

Whether you're new to coding or simply looking for a more efficient way to handle your data, Fast CSS Tool provides a user-friendly interface to help you get the job done.

![image](https://github.com/cssturkiye/fastcsstool/assets/53001810/c037c3fd-eb01-4d9e-8178-4c9e9b4fea59)


## Features

- **Data Preprocessing**: Import data, apply manual and AI-based filters.
- **Model Training**: Train multiple machine learning models using SentenceTransformers, Microsoft's FLAML and scikit-learn.
- **Model Evaluation**: Evaluate models using various metrics.
- **Export Results**: Save models and export evaluation results.

## Installation

### Prerequisites

- **Python 3.10 or later**
- **pip** (Python package installer)

### Windows Installation

1. **Download and Install Miniconda**:
   - Download the Miniconda installer for Windows from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).
   - Run the installer and follow the prompts to install Miniconda.

2. **Create and Activate Virtual Environment**:
   - **Open Command Prompt**:
     - Press `Win + R`, type `cmd`, and press `Enter`.
   - **Navigate to the Project Directory**:
     - Use the `cd` command to navigate to the directory where you have saved the Fast CSS Tool project. For example:
       ```batch
       cd fastcsstool
       ```
   - **Create a Virtual Environment**:
     - Run the following command to create a virtual environment named `fastcsstool-env`:
       ```batch
       conda create -y -p %cd%\\fastcsstool-env python=3.10
       ```
   - **Activate the Virtual Environment**:
     - Run the following command to activate the virtual environment:
       ```batch
       call fastcsstool-env\\Scripts\\activate
       ```

3. **Install PyTorch with GPU Support**:
   - Run the following command to install PyTorch with GPU support:
     ```batch
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

4. **Install Other Dependencies**:
   - Run the following command to install the required Python packages:
     ```batch
     pip install -r requirements.txt
     ```

5. **Run the Application**:
   - Ensure the virtual environment is activated.
   - Run the application by executing:
     ```batch
     python main.py
     ```

### Mac Installation

1. **Download and Install Miniconda**:
   - Download the Miniconda installer for Mac from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh).
   - Open Terminal and navigate to the directory where the installer is downloaded.
   - Run the installer with the following command:
     ```sh
     bash Miniconda3-latest-MacOSX-x86_64.sh
     ```
   - Follow the prompts to complete the installation.

2. **Create and Activate Virtual Environment**:
   - **Open Terminal**.
   - **Navigate to the Project Directory**:
     - Use the `cd` command to navigate to the directory where you have saved the Fast CSS Tool project. For example:
       ```sh
       cd fastcsstool
       ```
   - **Create a Virtual Environment**:
     - Run the following command to create a virtual environment named `fastcsstool-env`:
       ```sh
       conda create -y -p ./fastcsstool-env python=3.10
       ```
   - **Activate the Virtual Environment**:
     - Run the following command to activate the virtual environment:
       ```sh
       source fastcsstool-env/bin/activate
       ```

3. **Install PyTorch**:
   - Run the following command to install PyTorch:
     ```sh
     pip install torch torchvision torchaudio
     ```

4. **Install Other Dependencies**:
   - Run the following command to install the required Python packages:
     ```sh
     pip install -r requirements.txt
     ```

5. **Run the Application**:
   - Ensure the virtual environment is activated.
   - Run the application by executing:
     ```sh
     python main.py
     ```

## Usage

### Data Generation from Twitter

- **Bearer Token**: Enter your valid Twitter bearer token.
- **Keywords**: Set keywords to filter tweets.
- **Include Options**: Choose to include retweets and quotes.
- **Geo-Location**: Specify latitude, longitude, and radius for geographic targeting.
- **Date Range**: Select the start and end dates for your data collection.
- **Language**: Choose the language of the tweets to collect.
- **Search and Download**: Click to begin the data collection process.

### Data Labeling

- **Import CSV**: Load your data file for labeling.
- **Labels**: Input and update the labels for categorizing data.
- **Navigation**: Navigate through data entries and save your labeling progress.

### Data Preprocessing

- **Import Data**: Load your dataset.
- **Manual Filtering**: Apply filters like keyword exclusion and tweet length constraints.
- **AI-Based Filtering**: Use AI models to filter data automatically.
- **Export Data**: Save your filtered dataset for further analysis or training.

### Model Training & Evaluation

- **Training Data**: Load your dataset for model training.
- **Start Training**: Begin the training of your model.
- **Evaluation**: Assess the performance of your model with accuracy, recall, precision, and F1-score metrics.
- **Save Models**: Save your trained models for future use.

### Analyze Data

- **Import Data and Model**: Load your analysis model and dataset.
- **Graphical Analysis**: Perform and visualize various analyses like time series and distribution of data points.
- **Export Analysis Results**: Save your analysis results for reporting or documentation purposes.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE.

## Acknowledgements
This tool uses the following main packages
- [Sentence Transformers](https://sbert.net/) library for accessing, using, and training state-of-the-art text and image embedding models.
- [FLAML](https://microsoft.github.io/FLAML/) library for Automated Machine Learning & Tuning.
- [scikit-learn](https://scikit-learn.org/) library for machine learning algorithms.

For any issues or questions, please contact [info@csstr.org].
