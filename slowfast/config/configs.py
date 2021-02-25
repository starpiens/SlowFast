# Hyper-parameters
T = 4
alpha = 8
beta_inv = 8
blocks = (0, 0, 3, 4, 6, 3)

# Backbone
backbone = 'ResNet-18'
dim_inner = [0, 0]
dim_out = [0, 64]
if backbone == 'ResNet-18':
    dim_inner += [64, 128, 256, 512]
    dim_out += [64, 128, 256, 512]
elif backbone == 'ResNet-50':
    dim_inner += [64, 128, 256, 512]
    dim_out += [256, 512, 1024, 2048]

# Train
train_batch_size = 1
max_epoch = 192

# Dataset
num_classes = 400
dataset_path = "/Users/starlett/codes/my_slowfast/dataset/"

# Environment
num_gpus = 0