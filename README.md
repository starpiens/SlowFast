# SlowFast
PyTorch implementation of *SlowFast Networks for Video Recognition* ([arxiv](https://arxiv.org/abs/1812.03982)).

## Preparing Kinetics Dataset
Download dataset with [Kinetics downloader](https://github.com/jihun-kr/Kinetics-downloader).
For example, you could run:
```
cd Kinetics-downloader
python download.py data/kinetics-100-pruned_train.csv /data/kinetics-100/train/ -n 16 -t /data/kinetics-100/tmp/
```
Place `.csv` file under data directory, and rename as `train.csv` or `val.csv`.
You also need `classes.csv`.
```
cp Kinetics-downloader/data/kinetics-100-pruned_train.csv /data/kinetics-100/train.csv
cp Kinetics-downloader/data/kinetics-100-classes.csv /data/kinetics-100/classes.csv
```


## Setup
Create and start new Anaconda environment. 
```
conda create -n slowfast python=3.9
conda activate slowfast
```

Install pre-requisites.
```
pip install -r requirements.txt
```

Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/:$PYTHONPATH
```

## Run
Configure, and run training.
```
vi SlowFast/slowfast/
python tools/train.py
```
