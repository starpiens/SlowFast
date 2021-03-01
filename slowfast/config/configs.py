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
train_batch_size = 11
max_epoch = 192
num_gpus = 0  # Assuming 0 or 1.
checkpoint_path = "/Users/starlett/codes/my_slowfast/checkpoints/"

##### Dataset #####
num_classes = 100
dataset_path = "/Users/starlett/codes/my_slowfast/dataset/"
