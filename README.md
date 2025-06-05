
# Flask App Setup Guide

## Step 1: Create a Virtual Environment

Before running the Flask app, it is recommended to create a virtual environment to manage dependencies.

### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Required Packages

After activating the virtual environment, install all the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Step 3: Run the Flask Application

Finally, run the Flask app using the following command:

```bash
python app.py
```
