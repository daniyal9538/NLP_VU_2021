# Intro to NLP: Assignment 1

Authors: Jaimie R. Rutgers, Roberto Schiavone, Daniyal Selani

## Requirements
- Python 3.9.4
- CUDA 11.1
- an Internet connection

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
This script outputs all relevant results for part A of the assignment. For the
POS tagging, the script first prints the 10 most common POS tags with their
corresponding frequency. Next, the three most common and one of the least common
tokens is printed, per POS tag. For the lemmatization, the script outputs the
lemma and the token for the first 100 sentences. It requires manual work to
choose a lemma with three different inflections. 

### Part B
Executing the file as a python script will print out appropriate output to the
python console with explanatory messages, that relate directly to the
corresponding section in the report.

As an additional dependency, you need to install PyTorch (not included in the
`requirements.txt` because the CUDA version could not be specified there).

```
python -m pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Now, you can run:

```
python partB.py
```

### Part C
This script is just a modified version of the original BERT notebook, tweaked
to accomodate the needs of the task. The script will output one file for the
confusion matrix and one containing true and predicted labels.

As before, it is sufficient to run:

```
python partC.py
```

During the execution, the detailed level of logging will help with the
interpretation of every step, and at the end of the script every measure will
be output in the `stdout`.

