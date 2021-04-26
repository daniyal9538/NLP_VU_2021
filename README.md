# Intro to NLP: Assignment 1

Authors:

## Requirements
To keep intact the output of
```
python -m pip freeze > requirements.txt
```

we are listing here the hardware requirements:
- Python 3.9
- CUDA 11.1

## Environment
### (Optional) Create a virtual environment

```
python -m pip venv .venv
source .venv/bin/activate
```

### Setting things up
To run the code, it is necessary to install all the dependencies listed in 
`requirements.txt`:

```
python -m pip install -r requirements.txt
```

## Structure
### Part A

# COLA DATASET HERE

In addition to module dependencies, part A requires the download of the spaCy 
model **"en_core_web_sm"**. In case the download didn't start automatically by 
installing the requirements, run:

```
python -m spacy download en_core_web_sm
```

followed by

```
python partA.py
```

### Part B

Executing the file as a python script will print out appropriate output to the python console with explanatory messages, that relate directly to the corrosponding section in the report

```
python partB.py
```

### Part C

# IRONY DATASET AND TEST SET HERE

```
python partC.py
```

_Remarks: due to the file size of the datasets and the fine-tuned models, we
decided to attach only to code and the data required to reproduce our results,
without any additional models or outputs._

