huggingface-cli download facebook/opt-2.7b \
  --local-dir ./config/blip2-opt-2.7b/opt-2.7b \
  --local-dir-use-symlinks False

 huggingface-cli download lmsys/vicuna-7b-v1.3 \
   --local-dir ./config/blip2-vicuna-instruct-7b/vicuna-7b \
   --local-dir-use-symlinks False