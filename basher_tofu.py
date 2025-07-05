import os, random, argparse, time
import os, random, argparse, time, subprocess
parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--split', type=str,
                    help='forget01 forget05 forget10')
parser.add_argument('--model_family', type=str,
                    help='phi llama')

parser.add_argument('--cuda_id', type=int,
                    help='0~7')

args = parser.parse_args()

losses = []

gage = {"loss":"ga_ge","hyper_param":"none","gradient_norm":100,"moving_avg":0.6}
gage1 = {"loss":"ga_ge","hyper_param":"none","gradient_norm":100,"moving_avg":0.99}
losses = [gage,gage1]



forget_path = "forget_auto_tofu.py"


if args.model_family == "llama":
    config_name = "forget_llama.yaml"
    lr = 1e-5
    model_family = "llama2-7b"
if args.model_family == "phi":
    config_name = "forget_phi.yaml"
    lr = 2e-5
    model_family = "phi"



for param in losses:  
    log_file_path = f"{param['loss']}_{param['hyper_param']}_{param['gradient_norm']}_{param['moving_avg']}.log"
    command = (
        f"CUDA_VISIBLE_DEVICES={args.cuda_id} "
        f"torchrun --nproc_per_node=1 "
        f"--master_port={random.randint(10000, 60000)} "
        f"{forget_path} "
        f"--config-name={config_name} "
        f"split={args.split} "
        f"model_family={model_family} "
        f"forget_loss={param['loss']} "
        f"hyper_param={param['hyper_param']} "
        f"gru_type=proj "
        f"gradient_norm={param['gradient_norm']} "
        f"moving_avg={param['moving_avg']} "
        f"> {log_file_path}"
    )
    # os.system(command)
    # time.sleep(1000)

    # Start the process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Monitor the process output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            if "training done!" in output:  
                break

    # Ensure the process has finished before moving on
    process.wait()
    print(f"Finished processing: {param['loss']}")





