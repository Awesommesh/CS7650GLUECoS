# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NER_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}

EPOCH=5
BATCH_SIZE=16
MAX_SEQ=256

dir=`basename "$TASK"`
OUT=$(basename $(dirname "$TASK"))
fi

python $PWD/Code/BertToken.py \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUT_DIR/$OUT \
  --model_type $MODEL_TYPE \
  --model_name $MODEL \
  --num_train_epochs $EPOCH \
  --train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --num_layers 1 \
  --save_steps -1