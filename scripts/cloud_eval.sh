
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cxj
fi

cd ./AirVLN
echo $PWD

CONFIG_PATH=${CLOUD_EVAL_CONFIG:-configs/cloud_eval.yaml}
if [ -f configs/cloud_eval.local.yaml ]; then
  CONFIG_PATH=configs/cloud_eval.local.yaml
fi

python -u ./src/vlnce_src/cloud_eval.py \
--cloud_config "$CONFIG_PATH"
