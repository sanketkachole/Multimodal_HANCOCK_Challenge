import fire, hancock.pid_index as pid, hancock.fuse as fuse, hancock.train as trn

class CLI:
    def build_index(self, targets_csv='features/targets.csv', out='pid_index.pkl'):
        pid.build_pid_index(targets_csv, out)

    def fuse(self, config='config.yaml'):
        fuse.run(config)

    def train(self, model='xgb', config='config.yaml'):
        trn.run(model, config)

if __name__ == '__main__':
    fire.Fire(CLI)


"""
Run anything with a single, readable command:
python hancock_cli.py build_index
python hancock_cli.py fuse
python hancock_cli.py train --model xgb

"""


"""
# build index on login (small job)
python hancock_cli.py build_index

# submit heavy loaders in parallel
sbatch -J load_tma  --wrap="python hancock_cli.py fuse --step tma"
sbatch -J load_wsi  --gres=gpu:1 --wrap="python hancock_cli.py fuse --step wsi"

# after fuse is done
sbatch -J train_xgb --wrap="python hancock_cli.py train --model xgb"

"""