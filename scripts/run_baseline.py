"""第二周基线:在 ml-1m 上跑 SASRec,输出 valid / test 指标.

用法
----
    cd ~/genrec
    python scripts/run_baseline.py
    python scripts/run_baseline.py --model GRU4Rec --config config/gru4rec_ml1m.yaml
"""
import argparse
import functools
from logging import getLogger

import torch

# PyTorch 2.6 起 `torch.load` 的 `weights_only` 默认 True,而 RecBole 1.2.1 里是裸调用,
# 会在"训练结束后加载最优 checkpoint 做 test"这一步抛 UnpicklingError.
# checkpoint 是本进程自己产出的,改回 False 是安全的.
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load

from recbole.quick_start import run_recbole  # noqa: E402
from recbole.utils import dict2str  # noqa: E402


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='SASRec')
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--config', default='config/sasrec_ml1m.yaml')
    args = ap.parse_args()

    result = run_recbole(model=args.model, dataset=args.dataset,
                         config_file_list=[args.config])

    # run_recbole 内部只把 test 结果 logger.info 一次,不落盘,
    # 所以这里显式再打一遍,保证日志里有 test 那一行.
    logger = getLogger()
    logger.info('test result:\n%s', dict2str(result['test_result']))
    print('\n===== TEST RESULT =====')
    print(dict2str(result['test_result']))
