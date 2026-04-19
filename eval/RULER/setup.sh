pip install Cython
pip install -r requirements.txt
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2 --no-build-isolation
pip install nltk
python3 -c "import nltk; nltk.download('punkt')"
python3 -c "import nltk; nltk.download('punkt_tab')"

# prepare data
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh

cd ../../..