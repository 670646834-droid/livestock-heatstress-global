#!/usr/bin/env python3
"""Payback threshold sensitivity analysis on CPU1 (3/5/10 years)."""
import paramiko
import os
import csv

CPU1_HOST = "192.168.1.2"
CPU1_USER = "GBC"
CPU1_PASS = "gbc123456"
REMOTE_DIR = "/mnt/disk/GBC/livestock_heatstress"
LOCAL_SCRIPT = r"F:\20261011\project\02_Livestock_HeatStress\scripts\loss_estimation_v4_endogenized.py"
LOCAL_DATA_DIR = r"F:\20261011\project\02_Livestock_HeatStress\data_prep"

def main():
    print("=" * 70)
    print("Payback Threshold Sensitivity Analysis (3/5/10 years)")
    print("=" * 70)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CPU1_HOST, username=CPU1_USER, password=CPU1_PASS, timeout=30)
    print("Connected to CPU1.")

    # Upload script
    print("\nUploading v4 script (with --payback support)...")
    sftp = ssh.open_sftp()
    remote_script = f"{REMOTE_DIR}/loss_estimation_v4_endogenized_payback.py"
    sftp.put(LOCAL_SCRIPT, remote_script)
    sftp.close()
    print("Uploaded.")

    results = {}
    for payback in [3, 5, 10]:
        print(f"\n{'='*70}")
        print(f"Running payback={payback} years...")
        print(f"{'='*70}")

        cmd = f"cd {REMOTE_DIR} && python3 loss_estimation_v4_endogenized_payback.py --year 2024 --payback {payback} 2>&1"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode()
        print(output[-2000:] if len(output) > 2000 else output)

        # Download result
        remote_csv = f"{REMOTE_DIR}/data_prep/loss_estimation_v4_endogenized_2024_payback{payback}yr.csv"
        local_csv = os.path.join(LOCAL_DATA_DIR, f"loss_estimation_v4_endogenized_2024_payback{payback}yr.csv")
        print(f"\nDownloading to {local_csv}...")
        sftp = ssh.open_sftp()
        try:
            sftp.get(remote_csv, local_csv)
            print("Downloaded.")
            results[payback] = parse_csv(local_csv)
        except Exception as e:
            print(f"Download failed: {e}")
        sftp.close()

    ssh.close()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Payback Threshold Sensitivity (THI>=72, 2024)")
    print(f"{'='*70}")
    print(f"{'Payback':<10s} {'Gross($B)':<12s} {'Avoidable($B)':<16s} {'Avoid%':<10s} {'Residual($B)':<14s} {'Residual%':<10s}")
    print("-" * 72)
    for pb in [3, 5, 10]:
        if pb in results:
            r = results[pb]
            print(f"{pb}yr{'':<6s} {r['gross']:<12.2f} {r['avoidable']:<16.2f} {r['avoidable_pct']:<10.1f} {r['residual']:<14.2f} {r['residual_pct']:<10.1f}")

    print(f"\nAll CSVs saved to: {LOCAL_DATA_DIR}")

def parse_csv(csv_path):
    total_gross = total_avoidable = total_residual = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['threshold'] == 'thi_72':
                total_gross += float(row['gross_loss_usd'])
                total_avoidable += float(row['avoidable_loss_usd'])
                total_residual += float(row['residual_loss_usd'])
    return {
        'gross': total_gross / 1e9,
        'avoidable': total_avoidable / 1e9,
        'residual': total_residual / 1e9,
        'avoidable_pct': total_avoidable / total_gross * 100 if total_gross > 0 else 0,
        'residual_pct': total_residual / total_gross * 100 if total_gross > 0 else 0,
    }

if __name__ == "__main__":
    main()
