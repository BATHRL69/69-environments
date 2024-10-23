# Create virtual envionment
python -m venv venv

# PC: activate env, in cmd NOT POWERSHELL
venv\Scripts\activate

# install dependencies
pip install -r Walker/requirements.txt

# run the training
python Walker/Double_inverted_pendulum_learning.py