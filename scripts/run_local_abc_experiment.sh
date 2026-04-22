if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cxj
fi

CONFIG_PATH=${1:-configs/local_abc_experiment.yaml}
shift || true

cd ./AirVLN
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -u ./src/vlnce_src/run_local_abc_experiment.py "$CONFIG_PATH" "$@"
