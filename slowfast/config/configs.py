##### Model hyper-parameters. #####
T = 4
alpha = 8
beta_inv = 8
input_frames = 64

##### Model backbone. #####
backbone = 'ResNet-18'

if backbone == 'ResNet-18':
    dim_inner = [0, 0, 64, 128, 256, 512]
    dim_out = [0, 64, 64, 128, 256, 512]
    blocks = (0, 0, 2, 2, 2, 2)
elif backbone == 'ResNet-50':
    dim_inner = [0, 0, 64, 128, 256, 512]
    dim_out = [0, 64, 256, 512, 1024, 2048]
    blocks = (0, 0, 3, 4, 6, 3)

##### Train #####
train_batch_size = 16
cosine_lr_base = 1.0 * train_batch_size / 64
start_lr = 0.1 * train_batch_size / 64
warmup_epochs = 32
max_epoch = 192
num_gpus = 1  # Assuming 0 or 1.
checkpoint_path = "/data/SlowFast/slowfast/checkpoints/"

##### Dataset #####
num_classes = 100
dataset_path = "/data/Kinetics-100/"
