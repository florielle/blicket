import subprocess

i = 1

for d_embed in [30, 50]:
    for hidden_size in [15, 20, 25]:
        for e_dropout in [0.2, 0.3, 0.4]:
            for d_dropout in [0.2, 0.3, 0.4]:
                for n_layers in [1,2]:
                    for tf_ratio in [0.08, .1, 0.12]:
                        for bidir in ['--bidirectional','']:
                            slurm_text = """#!/bin/bash
#
#SBATCH --job-name=blicket
#SBATCH --time=40:00:00
#SBATCH --mem=500MB
#SBATCH --output=outputs/exp_%A.out
#SBATCH --error=outputs/exp_%A.err

cd /scratch/jmw784/advancing_ai/

module purge
module load python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4
python3 -m pip install https://github.com/pytorch/text/archive/master.zip --user --upgrade

python3 -u main.py --d_embed {} --hidden_size {} --e_dropout {} --d_dropout {} --n_layers {} --tf_ratio {} {} --experiment {} > logs/{}.log
""".format(d_embed, hidden_size, e_dropout, d_dropout, n_layers, tf_ratio, bidir, i, i)

                            text_file = open("temp.slurm", "w")
                            text_file.write("%s" % slurm_text)
                            text_file.close()

                            subprocess.call("sbatch ./temp.slurm", shell=True)
                            i += 1
