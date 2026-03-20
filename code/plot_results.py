import re
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Read timing results the results.txt
# Expected lines printed by your CUDA program:
# N = 100000
# CPU merge time : 0.012345 s
# GPU merge time : 0.003210 s (kernel only)
# Speedup (CPU/GPU): 3.84 x
# ------------------------------------------------------------


RESULT_FILE = "results.txt"

Ns = []
cpu_times = []
gpu_times = []
speedups = []

with open(RESULT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        # Match N
        m = re.match(r"^N\s*=\s*(\d+)", line)
        if m:
            Ns.append(int(m.group(1)))
            continue

        # Match CPU merge time
        m = re.match(r"^CPU merge time\s*:\s*([0-9.eE+-]+)", line)
        if m:
            cpu_times.append(float(m.group(1)))
            continue

        # Match GPU merge time
        m = re.match(r"^GPU merge time\s*:\s*([0-9.eE+-]+)", line)
        if m:
            gpu_times.append(float(m.group(1)))
            continue

        # Match Speedup
        m = re.match(r"^Speedup.*:\s*([0-9.eE+-]+)", line)
        if m:
            speedups.append(float(m.group(1)))
            continue

# ------------------------------------------------------------
# Safety Check
# ------------------------------------------------------------
if not (len(Ns) == len(cpu_times) == len(gpu_times) == len(speedups)):
    print("Error: Parsed data lengths do not match!")
    print("Ns:", Ns)
    print("CPU:", cpu_times)
    print("GPU:", gpu_times)
    print("Speedup:", speedups)
    exit(1)

# ------------------------------------------------------------
# Plot 1: CPU Time
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(Ns, cpu_times, marker='o')
plt.xlabel("Input size N")
plt.ylabel("CPU time (seconds)")
plt.title("CPU Merge Time vs N")
plt.grid(True)
plt.tight_layout()
plt.savefig("cpu_time_vs_n.png")
plt.show()

# ------------------------------------------------------------
# Plot 2: GPU Time
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(Ns, gpu_times, marker='o', color='green')
plt.xlabel("Input size N")
plt.ylabel("GPU time (seconds)")
plt.title("GPU Merge Time vs N (Kernel Only)")
plt.grid(True)
plt.tight_layout()
plt.savefig("gpu_time_vs_n.png")
plt.show()

# ------------------------------------------------------------
# Plot 3: Speedup
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(Ns, speedups, marker='o', color='red')
plt.xlabel("Input size N")
plt.ylabel("Speedup (CPU time / GPU time)")
plt.title("Speedup vs N")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_vs_n.png")
plt.show()

print("Plots generated successfully.")