lr=2e-5
fuse_lr=2e-5
image_output_type=all
optim=adamw
data_type=MVSA-multiple
train_fuse_model_epoch=10
epoch=20
warmup_step_epoch=2
text_model=bert-base
fuse_type=att
batch=8
acc_grad=1
tran_num_layers=4
image_num_layers=2

python main.py -cuda -gpu_num $1 -epoch ${epoch} -num_workers 8 \
        -add_note ${data_type}-${tran_num_layers}-${fuse_type}-${batch}-${image_num_layers}-${text_model} \
        -data_type ${data_type} -text_model ${text_model} -image_model convnext_base \
        -batch_size ${batch} -acc_grad ${acc_grad} -fuse_type ${fuse_type} -image_output_type ${image_output_type} -fixed_image_model \
        -data_path_name 10-flod-1 -optim ${optim} -warmup_step_epoch ${warmup_step_epoch} -lr ${lr} -fuse_lr ${fuse_lr} \
        -tran_num_layers ${tran_num_layers} -image_num_layers ${image_num_layers} -train_fuse_model_epoch ${train_fuse_model_epoch} \
        -my_type 2
        
# 0 bert
# 1 convnext
# 2 根据MVSA数据集训练预训练模型
# 3 仅图像
# 4 仅文本
# 5 直接训练
# 6 根据MVSA预训练模型进行微调