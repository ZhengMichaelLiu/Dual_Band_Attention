cd "$(dirname "$0")/../eval/RULER"
bash setup.sh
cd scripts

echo "Running RULER: metric = Full attention"
bash run.sh llama3.1-8b-instruct synthetic --metric full

echo "Running RULER: metric = FlexPrefill"
bash run.sh llama3.1-8b-instruct synthetic --metric flex

echo "Running RULER: metric = MInference"
bash run.sh llama3.1-8b-instruct synthetic --metric minfer

echo "Running RULER: metric = XAttention, stride = 8"
bash run.sh llama3.1-8b-instruct synthetic --metric xattn --stride 8

echo "Running RULER: metric = XAttention, stride = 16"
bash run.sh llama3.1-8b-instruct synthetic --metric xattn --stride 16

echo "Running RULER: metric = DBA, stride = 8"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 8

echo "Running RULER: metric = DBA, stride = 16"
bash run.sh llama3.1-8b-instruct synthetic --metric dba --stride 16

echo "RULER finished."
