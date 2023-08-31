SERVICE_PATH=`pwd`

export PYTHONPATH=${PYTHONPATH}:${SERVICE_PATH}

python examples/PAI_QA_robot.py \
    --config ./config.json \
    --prompt_engineering "Retrieval-Augmented Generation" \
    --embed_model "SGPT-125M-weightedmean-nli-bitfit" \
    --embed_dim 768 \
    --upload \
    --query "什么是流式计算？请详细向我解释" \
    |& tee output.log
