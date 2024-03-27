import os
import json
import sys
from loguru import logger

CACHE_DIR = 'cache/'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CACHE_CONFIG_NAME = 'config.json'

def check_db_cache(keys, new_config):
    cache_path = os.path.join(CACHE_DIR, CACHE_CONFIG_NAME)
    # Check if there is local cache for bm25
    if not os.path.exists(cache_path):
        with open(cache_path, 'w+') as f:
            json.dump(new_config, f)
        return False

    # Read cached config file
    with open(cache_path, 'r') as f:
        cache_config = json.load(f)
    # Check if new_config is consistent with cache_config
    res = all([cache_config[k]==new_config[k] for k in keys])

    # Update cached config file
    with open(cache_path, 'w+') as f:
        json.dump(new_config, f)

    return res

emb_dim_dict = {
    "SGPT-125M-weightedmean-nli-bitfit": 768,
    "text2vec-large-chinese": 1024,
    "text2vec-base-chinese": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384
}
# def connect_adb(service, _global_args, _global_cfg, env_params, emb_model, emb_dim, emb_openai_key, llm_src, eas_url, eas_token, open_api_key, pg_host, pg_user, pg_pwd, pg_database, pg_collection, pg_del):
#     cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
#     cfg_db = {
#             'embedding': {
#                 "embedding_model": env_params['EMB_MODEL_NAME'],
#                 "embedding_dimension": emb_dim_dict[env_params['EMB_MODEL_NAME']],
#             },
#             'ADBCfg': {
#                 "PG_HOST": pg_host,
#                 "PG_DATABASE": pg_database,
#                 "PG_USER": pg_user,
#                 "PG_PASSWORD": pg_pwd,
#                 "PG_COLLECTION_NAME": pg_collection,
#                 "PRE_DELETE": pg_del
#             },
#         }
#     cfg.update(cfg_db)
#     _global_args.vectordb_type = "AnalyticDB"
#     _global_cfg.update(cfg)
#     _global_args.bm25_load_cache = check_db_cache(['vector_store', 'ADBCfg'], _global_cfg)
#     service.init_with_cfg(_global_cfg, _global_args)
#     return "Connect AnalyticDB success."

# def connect_holo(emb_model, emb_dim, emb_openai_key, llm_src, eas_url, eas_token, open_api_key, pg_host, pg_database, pg_user, pg_pwd, table):
#     cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
#     cfg_db = {
#         'embedding': {
#             "embedding_model": emb_model,
#             "model_dir": "./embedding_model/",
#             "embedding_dimension": emb_dim,
#             "openai_key": emb_openai_key
#         },
#         'EASCfg': {
#             "url": eas_url,
#             "token": eas_token
#         },
#         'HOLOCfg': {
#             "PG_HOST": pg_host,
#             "PG_DATABASE": pg_database,
#             "PG_PORT": 80,
#             "PG_USER": pg_user,
#             "PG_PASSWORD": pg_pwd,
#             "TABLE": table
#         },
#         "create_docs":{
#             "chunk_size": 200,
#             "chunk_overlap": 0,
#             "docs_dir": "docs/",
#             "glob": "**/*"
#         }
#     }
#     cfg.update(cfg_db)
#     _global_args.vectordb_type = "Hologres"
#     _global_cfg.update(cfg)
#     _global_args.bm25_load_cache = check_db_cache(['vector_store', 'HOLOCfg'], _global_cfg)
#     service.init_with_cfg(_global_cfg, _global_args)
#     return "Connect Hologres success."

# def connect_es(emb_model, emb_dim, emb_openai_key, llm_src, eas_url, eas_token, open_api_key, es_url, es_index, es_user, es_pwd):
#     cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
#     cfg_db = {
#         'embedding': {
#             "embedding_model": emb_model,
#             "model_dir": "./embedding_model/",
#             "embedding_dimension": emb_dim,
#             "openai_key": emb_openai_key
#         },
#         'EASCfg': {
#             "url": eas_url,
#             "token": eas_token
#         },
#         'ElasticSearchCfg': {
#             "ES_URL": es_url,
#             "ES_INDEX": es_index,
#             "ES_USER": es_user,
#             "ES_PASSWORD": es_pwd
#         },
#         "create_docs":{
#             "chunk_size": 200,
#             "chunk_overlap": 0,
#             "docs_dir": "docs/",
#             "glob": "**/*"
#         }
#     }
#     cfg.update(cfg_db)
#     _global_args.vectordb_type = "ElasticSearch"
#     _global_cfg.update(cfg)
#     _global_args.bm25_load_cache = check_db_cache(['vector_store', 'ElasticSearchCfg'], _global_cfg)
#     service.init_with_cfg(_global_cfg, _global_args)
#     return "Connect ElasticSearch success."

def connect_faiss(service, _global_args, _global_cfg, env_params):
    cfg_db = {
        'embedding': {
            "embedding_model": env_params['EMB_MODEL_NAME'],
            "embedding_dimension": emb_dim_dict[env_params['EMB_MODEL_NAME']],
        },
        'LLM': 'EAS',
        'EASCfg': {
            "url": env_params['EAS_URL'],
            "token": ''
        },
        "FAISS": {
            "index_path": env_params['FAISS_PATH'],
            "index_name": env_params['FAISS_INDEX'],
        }
    }
    _global_args.vectordb_type = "FAISS"
    _global_cfg.update(cfg_db)
    _global_args.bm25_load_cache = check_db_cache(['vector_store', 'FAISS'], _global_cfg)
    logger.info("Starting connecting to Local Faiss VectorStore.")
    service.init_with_cfg(_global_cfg, _global_args)

# def connect_milvus(emb_model, emb_dim, emb_openai_key, llm_src, eas_url, eas_token, open_api_key, milvus_collection, milvus_host, milvus_port, milvus_user, milvus_pwd, milvus_drop):
#     cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key,)
#     cfg_db = {
#         'embedding': {
#             "embedding_model": emb_model,
#             "model_dir": "./embedding_model/",
#             "embedding_dimension": emb_dim,
#             "openai_key": emb_openai_key
#         },
#         'MilvusCfg': {
#             "COLLECTION": milvus_collection,
#             "HOST": milvus_host,
#             "PORT": milvus_port,
#             "USER": milvus_user,
#             "PASSWORD": milvus_pwd,
#             "DROP": milvus_drop
#         },
#         "create_docs":{
#             "chunk_size": 200,
#             "chunk_overlap": 0,
#             "docs_dir": "docs/",
#             "glob": "**/*"
#         }
#     }
#     cfg.update(cfg_db)
#     _global_args.vectordb_type = "Milvus"
#     _global_cfg.update(cfg)
#     _global_args.bm25_load_cache = check_db_cache(['vector_store', 'MilvusCfg'], _global_cfg)
#     service.init_with_cfg(_global_cfg, _global_args)
#     return "Connect Milvus success."
