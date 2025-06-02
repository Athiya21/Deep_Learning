# Deep_Learning

# Steps to Install Deep Learning Framework using Anaconda
  - open anaconda cmd prompt
  - Create a new environment:
      conda create -n tensorflow_env tensorflow
  - Activate the environment:
      activate tensorflow_env
  - open anaconda navigator
  - jupyter and spyder installation (latest version) in anaconda navigator
  - After installation, open Anaconda Command Prompt again Install required libraries:
      - pip show tensorflow
      - pip install --upgrade tensorflow-cpu
      - pip install keras
      - pip install keras --upgrade
      - pip show keras
      - pip install torch
      - pip show torch
      - pip install ultralytics (yolo)
      - pip install langchain
      - pip install opencv-python
      - pip install mediapipe
      - pip install scikit-learn nltk spacy gensim numpy pandas matplotlib seaborn
        # after installation you must need to test
            import tensorflow as tf
            tf.__version__

# Steps to Install Deep Learning Framework in VS Code
  - create new folder
  - Open terminal and navigate to the folder (e.g: cd Deeplearning)
  - Create a virtual environment:
      python -m venv dlenv
  - If you created the environment using **Conda**:
      conda activate dlenv
  - If you created the environment using virtualenv:
      dlenv\Scripts\activate  # On Windows
  - Install required libraries:
    - conda install --forge spacy
    - pip install tensorflow
    - pip show tensorflow
    - pip install --upgrade tensorflow-cpu
    - pip install keras
    - pip install keras --upgrade
    - pip show keras
    - pip install torch
    - pip show torch
    - pip install ultralytics (yolo)
    - pip install langchain
    - pip install opencv-python
    - pip install mediapipe
    - pip install scikit-learn nltk spacy gensim numpy pandas matplotlib seaborn
