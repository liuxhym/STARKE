## Subset analysis of regularized kernel learners (Starke)

[ICML 22] Official implementation of paper: The Teaching Dimension of Regularized Kernel Learners

## Install
```bash
# Clone the code to local
git clone https://github.com/liuxhym/STARKE.git
cd STARKE

# Create virtual environment
conda create -n starke python=3.6
conda activate starke

# Install basic dependency
pip install -r requirements.txt
```

## Main Experiment
To reproduce the main experiment reported in paper, run following scripts:
```bash
python STARKE/example.py
```

## Reference

```latex
@inproceedings{
  qian2022teaching,
  title={The Teaching Dimension of Regularized Kernel Learners},
  author={Hong Qian and Xu-Hui Liu and Chenxi Su and Aimin Zhou and Yang Yu},
  booktitle={Proceedings of the 39th International Conference on Maching Learning},
  year={2022}
}
```
