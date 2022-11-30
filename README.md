
# DQN for Breakout

This project is the implementation for Natural and Dueling DQN for Breakout.

## Usage

```shell
pip install -r requirements.freezed
./run_dueling.sh
```

You can modify the parameters according to your own needs and refer to `config.py` for details. 

## Results
We have trained Dueling DQN for 50m steps and we evaluated the reward value every 0.1m steps. We obtained the best model with reward 795.3 at step 421. 


## References
On the basis of https://gitee.com/goluke/dqn-breakout#dqn-for-breakout, we mainly modify `main.py`, `utils_drl.py`, `utils_model.py`, `run.sh` and add `config.py`, `run_dueling.sh`, `run_pretrain.sh`.


## Folder Structures
We only explain the contents in the folder dqn.The rest is similar to it.
```bash
.
├── dqn         #普通dqn
│   ├── calc.py     #添加时间阈值，超过阈值的直接退出循环
│   ├── check.sh
│   ├── display.ipynb   #结果导出为视频，只需要修改tarjet即可选择导出的模型。
│   ├── eval_372
│   ├── main.py     
│   ├── models_1    #model每训练10W轮保存的模型。
│   ├── __pycache__
│   ├── README.md
│   ├── requirements.freezed
│   ├── result.txt  #重新运行的结果保存在result.txt中
│   ├── rewards.txt #原始输出数据
│   ├── run.sh      #运行只需要执行 CUDA_VISIBLE_DEVICES=X sh run.sh 即可，其中X是显卡编号。
│   ├── tmp_require
│   ├── utils_drl.py
│   ├── utils_env.py
│   ├── utils_memory.py
│   ├── utils_model.py
│   ├── utils_types.py
│   └── vendor
├── dqn+sum_tree    #加上sum_tree经验回放（不过没有调出来）
│   ├── check.sh
│   ├── display.ipynb
│   ├── eval_390
│   ├── main.py
│   ├── models
│   ├── __pycache__
│   ├── README.md
│   ├── requirements.freezed
│   ├── rewards.txt
│   ├── run.sh
│   ├── SumTree.py
│   ├── utils_drl.py
│   ├── utils_env.py
│   ├── utils_memory.py
│   ├── utils_model.py
│   ├── utils_types.py
│   └── vendor
├── dueling_dqn #Dueling_dqn
│   ├── calc.py
│   ├── check.sh
│   ├── display.ipynb
│   ├── main.py
│   ├── models
│   ├── models_1
│   ├── __pycache__
│   ├── README.md
│   ├── requirements.freezed
│   ├── result.txt
│   ├── rewards.txt
│   ├── run.sh
│   ├── tmp_eval_frames
│   ├── utils_drl.py
│   ├── utils_env.py
│   ├── utils_memory.py
│   ├── utils_model.py
│   ├── utils_types.py
│   └── vendor
└── README.md