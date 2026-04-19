cd "$(dirname "$0")/../eval/RULER"
bash setup.sh
cd scripts

echo "Running RULER: metric = DBA, stride = 16, threshold = 0.7"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.7

echo "Running RULER: metric = DBA, stride = 16, threshold = 0.8"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.8


echo "Running RULER: metric = DBA, stride = 16, threshold = 0.85"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.85

echo "Running RULER: metric = DBA, stride = 16, threshold = 0.9"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.9


echo "Running RULER: metric = DBA, stride = 16, threshold = 0.95"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.95

echo "Running RULER: metric = DBA, stride = 16, threshold = 0.97"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16 --threshold 0.97


echo "RULER finished."
