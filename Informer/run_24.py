import os

# Make sure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Base command with common settings
base_command = (
    "python -u main_informer.py "
    "--model informer "
    "--data ETTh1 "
    "--features M "
    "--e_layers 2 "
    "--d_layers 1 "
    "--attn prob "
    "--des 'Exp' "
    "--itr 5 "
)

# List of (seq_len, label_len, pred_len) combinations
configs = [
    (48, 24, 24),
    (48, 12, 24),
    (24, 12, 24),
    (24, 6, 24),
    (96, 96, 24),
    (96, 48, 24),
    (96, 24, 24),
]

# Loop through configurations and run the experiments
for seq_len, label_len, pred_len in configs:
    log_file = f"logs/pred24/seq{seq_len}_label{label_len}_pred{pred_len}.txt"
    command = (
        f"{base_command} "
        f"--seq_len {seq_len} "
        f"--label_len {label_len} "
        f"--pred_len {pred_len} "
        f"> {log_file}"
    )
    print(f"\n▶️ Running: seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}")
    print(f"   ➜ Logging output to: {log_file}")
    os.system(command)
