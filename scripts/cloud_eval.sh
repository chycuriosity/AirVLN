
conda activate cxj

cd ./AirVLN
echo $PWD

python -u ./src/vlnce_src/cloud_eval.py \
--run_type eval \
--collect_type TF \
--name Cloud-Qwen35-Flash \
--batchSize 1 \
--EVAL_DATASET val_unseen \
--EVAL_NUM 1 \
--maxAction 20 \
--cloud_model qwen3.5-flash \
--cloud_use_depth_summary
