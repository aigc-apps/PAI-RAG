def auto_configure_chatglm_device_map(num_gpus: int):
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_map = {
        "transformer.word_embeddings": 0,
        "transformer.final_layernorm": 0,
        "lm_head": 0,
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f"transformer.layers.{i}"] = gpu_target
        used += 1

    return device_map