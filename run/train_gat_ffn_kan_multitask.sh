# python train_gat_ffn_kan_multitask.py \
#   --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification \
#   --save_folder ./save/graph_attention_ffn_kan_multitask \
#   --model_name graph_attention_ffn_kan_multitask \
#   --batch_size 64 \
#   --num_workers 8 \
#   --epochs 100 \
#   --print_freq 100 \
#   --learning_rate 0.001 \
#   --weight_decay 0.0001 \
#   --hidden_dim 128 \
#   --num_layers 3 \
#   --heads 4 \
#   --id_emb_dim 32 \
#   --rel_emb_dim 8 \
#   --dropout 0.2 \
#   --kan_hidden 128 \
#   --loss_name ce \
#   --use_class_weights \
#   --use_node_class_weights \
#   --enable_node_task \
#   --node_loss_weight 1.0 \
#   --selection_metric joint \
#   --node_target node_y \
#   --kan_reg_lambda 0.00001 \
#   --device cuda > ./save/log/train_gat_ffn_kan_multitask.log 2>&1 &


python train_gat_ffn_kan_multitask_updated.py \
  --graph_folder ./data/2017-subaru-forester/graphs_subsample_node_classification \
  --save_folder ./save/graph_attention_ffn_kan_multitask_v2 \
  --model_name graph_attention_ffn_kan_multitask_v2 \
  --batch_size 64 \
  --num_workers 8 \
  --epochs 100 \
  --print_freq 100 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --hidden_dim 128 \
  --num_layers 3 \
  --node_head_from_layer 0 \
  --heads 4 \
  --id_emb_dim 32 \
  --rel_emb_dim 8 \
  --dropout 0.2 \
  --kan_hidden 128 \
  --loss_name ce \
  --use_class_weights \
  --enable_node_task \
  --node_loss_weight 1.0 \
  --selection_metric joint \
  --node_target node_y \
  --kan_reg_lambda 0.00001 \
  --device cuda \
  --save_epoch_checkpoints \
  --epoch_save_every 10 \
  --print_val_node_cm_every 5 \
  --save_val_node_cm \
  > ./save/log/train_gat_ffn_kan_multitask_no_class_weights.log 2>&1 &