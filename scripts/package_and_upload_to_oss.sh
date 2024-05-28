timestamp=$(date +"%Y%m%d-%H%M%S")-$(date +%N | cut -c1-3)

tar -czvf EasyRAG_${timestamp}.tar.gz data/custom_dataset docker package scripts src poetry.lock pyproject.toml README.md
ossutil cp EasyRAG_${timestamp}.tar.gz oss://pai-rag/codes/
ossutil set-acl oss://pai-rag/codes/EasyRAG_${timestamp}.tar.gz public-read
ossutil ls oss://pai-rag/codes/EasyRAG_${timestamp}.tar.gz
