import argparse
import os
import datetime
from pathlib import Path
import yaml
from utils.CN import CN


CONFIG_SECTION_PREFIXES = {
    "cloud": "cloud_",
    "eval": "",
    "simulator": "",
}


def _flatten_config(config, parent_key=None):
    flattened = {}
    for key, value in config.items():
        if isinstance(value, dict):
            prefix = CONFIG_SECTION_PREFIXES.get(key, "{}_".format(key))
            nested = _flatten_config(value, key)
            for nested_key, nested_value in nested.items():
                flattened["{}{}".format(prefix, nested_key)] = nested_value
        else:
            flattened[key] = value
    return flattened


def _load_config_file(config_path):
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(str(os.getcwd())).resolve() / path

    with open(str(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping: {}".format(path))

    return _flatten_config(data)


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        project_prefix = Path(str(os.getcwd())).parent.resolve()
        self.parser.add_argument('--cloud_config', type=str, default=None, help="cloud evaluation YAML config path")
        self.parser.add_argument('--project_prefix', type=str, default=str(project_prefix), help="project path")

        self.parser.add_argument('--run_type', type=str, default="train", help="run_type in [collect, train, eval]")
        self.parser.add_argument('--policy_type', type=str, default="seq2seq", help="policy_type in [seq2seq, cma]")
        self.parser.add_argument('--collect_type', type=str, default="TF", help="seq2seq in [TF, dagger, SF]")
        self.parser.add_argument('--name', type=str, default='default', help='experiment name')

        self.parser.add_argument('--maxInput', type=int, default=300, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=500, help='max action sequence')

        self.parser.add_argument("--dagger_it", type=int, default=1)
        self.parser.add_argument("--epochs", type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.00025, help="learning rate")
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument("--trainer_gpu_device", type=int, default=0, help='GPU')

        self.parser.add_argument('--Image_Height_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Width_RGB', type=int, default=224)
        self.parser.add_argument('--Image_Height_DEPTH', type=int, default=256)
        self.parser.add_argument('--Image_Width_DEPTH', type=int, default=256)

        self.parser.add_argument('--inflection_weight_coef', type=float, default=1.9)

        self.parser.add_argument('--nav_graph_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/nav_graph_10'), help="nav_graph path")
        self.parser.add_argument('--token_dict_path', type=str, default=str(project_prefix / 'DATA/data/disceret/processed/token_dict_10'), help="token_dict path")
        self.parser.add_argument('--vertices_path', type=str, default=str(project_prefix / 'DATA/data/disceret/scene_meshes'))
        self.parser.add_argument('--dagger_mode_load_scene', nargs='+', default=[])
        self.parser.add_argument('--dagger_update_size', type=int, default=8000)
        self.parser.add_argument('--dagger_mode', type=str, default="end", help='dagger mode in [end middle nearest]')
        self.parser.add_argument('--dagger_p', type=float, default=1.0, help='dagger p')

        self.parser.add_argument('--TF_mode_load_scene', nargs='+', default=[])

        self.parser.add_argument('--ablate_instruction', action="store_true")
        self.parser.add_argument('--ablate_rgb', action="store_true")
        self.parser.add_argument('--ablate_depth', action="store_true")
        self.parser.add_argument('--SEQ2SEQ_use_prev_action', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_use', action="store_true")
        self.parser.add_argument('--PROGRESS_MONITOR_alpha', type=float, default=1.0)

        self.parser.add_argument('--EVAL_CKPT_PATH_DIR', type=str)
        self.parser.add_argument('--EVAL_DATASET', type=str, default="val_unseen")
        self.parser.add_argument("--EVAL_NUM", type=int, default=-1)
        self.parser.add_argument('--EVAL_GENERATE_VIDEO', action="store_true")

        self.parser.add_argument('--cloud_model', type=str, default="qwen3.5-flash")
        self.parser.add_argument('--cloud_base_url', type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.parser.add_argument('--cloud_api_key', type=str, default=None)
        self.parser.add_argument('--cloud_api_key_env', type=str, default="DASHSCOPE_API_KEY")
        self.parser.add_argument('--cloud_temperature', type=float, default=0.0)
        self.parser.add_argument('--cloud_max_tokens', type=int, default=64)
        self.parser.add_argument('--cloud_timeout', type=float, default=60.0)
        self.parser.add_argument('--cloud_max_retries', type=int, default=3)
        self.parser.add_argument('--cloud_fallback_action', type=str, default="STOP")
        self.parser.add_argument('--cloud_no_rgb', action="store_true")
        self.parser.add_argument('--cloud_use_depth_summary', action="store_true")
        self.parser.add_argument('--cloud_save_prompts', action="store_true")

        self.parser.add_argument('--rgb_encoder_use_place365', action="store_true")
        self.parser.add_argument('--tokenizer_use_bert', action="store_true")

        self.parser.add_argument("--simulator_tool_port", type=int, default=30000, help="simulator_tool port")
        self.parser.add_argument("--DDP_MASTER_PORT", type=int, default=20000, help="DDP MASTER_PORT")

        self.parser.add_argument("--continue_start_from_dagger_it", type=int)
        self.parser.add_argument("--continue_start_from_checkpoint_path", type=str)

        self.parser.add_argument('--vlnbert', action="store_true", default="prevalent")
        self.parser.add_argument('--featdropout', action="store_true", default=0.4)
        self.parser.add_argument('--action_feature', action="store_true", default=32)

        self.args = self.parser.parse_args()

        if self.args.cloud_config:
            config_values = _load_config_file(self.args.cloud_config)
            valid_keys = {action.dest for action in self.parser._actions}
            invalid_keys = sorted([key for key in config_values if key not in valid_keys])
            if invalid_keys:
                raise KeyError(
                    "Unsupported config keys in {}: {}".format(
                        self.args.cloud_config,
                        ", ".join(invalid_keys),
                    )
                )
            for key, value in config_values.items():
                setattr(self.args, key, value)


param = Param()
args = param.args

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/DATA/output/{}/{}/logs/{}_{}.log'.format(args.project_prefix, args.name, args.run_type, args.name, args.make_dir_time)


# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval'], 'run_type error'
# args.policy_type = 'seq2seq'
assert args.policy_type in ['seq2seq', 'cma'], 'policy_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'


args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]


args.TRAIN_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038


default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()
