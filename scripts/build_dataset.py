"""
Build the final provider-agnostic modeling dataset (vCPU / RAM instead of
anonymized provider/family codes) from the IBM multi-cloud-configuration
dataset.

Decoding method: see scripts/decode_pricing.py. The anonymized labels
were resolved by recovering the exact implied hourly price per config
(cost = runtime_h * price * nodes, std ~ 0 within each group) and
matching the price ordering against known real-world instance pricing
hierarchies (e.g. c4 < m4 < r4 for AWS; e2 < n1 for GCP; highcpu <
standard < highmem for GCP types; D_v3 < D_v2 for Azure due to
hyperthreaded vCPUs in v3).

Design decision: only provider C (GCP) is kept.
  - GCP has by far the most rows (1486 raw / 1282 status==ok), vs.
    A=737/664 and B=284/256.
  - Critically, within GCP alone every (vcpu, ram_gb) pair maps to
    exactly ONE real instance type (0/1237 collisions), because GCP's
    12 instance types (e2/n1 x standard/highcpu/highmem x 2/4 vCPU)
    all have distinct RAM amounts. This means vcpu+ram alone are a
    lossless, deterministic stand-in for "which instance" - cost and
    runtime become clean, noise-free targets.
  - Mixing all 3 providers and dropping the provider label instead
    causes ~12% of (vcpu,ram,nodes,workload) combos to collide across
    providers with wildly different target_cost (mean relative spread
    ~140%, since cost is dominated by provider pricing policy, not
    hardware). GCP-only avoids this entirely.

Resulting mapping (documented in full in README_mapping.md):

  C_family: 0=e2,  1=n1           C_type: 0=standard, 1=highcpu, 2=highmem
  C_vcpu:   0=2vCPU, 1=4vCPU
"""
import pandas as pd

df = pd.read_csv("data/multi-cloud-configuration.csv", index_col=0)
df = df[df["provider"] == "C"].copy()

GCP = {  # (C_family, C_type) -> {C_vcpu: (vcpu, ram_gb)}
    (0, 0): {0: (2, 8.0), 1: (4, 16.0)},    # e2-standard
    (0, 1): {0: (2, 2.0), 1: (4, 4.0)},     # e2-highcpu
    (0, 2): {0: (2, 16.0), 1: (4, 32.0)},   # e2-highmem
    (1, 0): {0: (2, 7.5), 1: (4, 15.0)},    # n1-standard
    (1, 1): {0: (2, 1.8), 1: (4, 3.6)},     # n1-highcpu
    (1, 2): {0: (2, 13.0), 1: (4, 26.0)},   # n1-highmem
}
GCP_TYPE_NAME = {0: "standard", 1: "highcpu", 2: "highmem"}


def decode_row(row):
    fam, typ, v = int(row["C_family"]), int(row["C_type"]), int(row["C_vcpu"])
    vcpu, ram = GCP[(fam, typ)][v]
    fam_name = "e2" if fam == 0 else "n1"
    instance = f"{fam_name}-{GCP_TYPE_NAME[typ]}-{vcpu}"
    return pd.Series({"vcpu": vcpu, "ram_gb": ram, "instance_type": instance})


decoded = df.apply(decode_row, axis=1)
out = pd.concat([df, decoded], axis=1)

# split workload into (dataset, task) -- format is "<dataset>-<task>"
out[["dataset", "task"]] = out["workload"].str.split("-", n=1, expand=True)

# keep only successful runs: failed/timed-out rows have unreliable
# runtime/cost measurements (see original dataset README)
out = out[out["status"] == "ok"].copy()

final_cols = [
    "vcpu", "ram_gb", "nodes", "dataset", "task",
    "target_cost", "target_runtime",
    "instance_type",  # kept only for reference/debugging, not for modeling
]
out = out[final_cols].reset_index(drop=True)
out.to_csv("data/cloud_cost_dataset_gcp.csv", index=False)

print(out.head(10).to_string())
print("\nShape:", out.shape)
print("\nUnique instance types:", out["instance_type"].nunique())
print("Unique (dataset, task) workloads:", out[["dataset", "task"]].drop_duplicates().shape[0])
print("Unique nodes values:", sorted(out["nodes"].unique().tolist()))

# sanity check: (vcpu, ram_gb) uniquely determines instance_type
check = out.groupby(["vcpu", "ram_gb"])["instance_type"].nunique()
assert (check == 1).all(), "collision found: vcpu+ram_gb does not uniquely identify instance_type"
print("\nOK: (vcpu, ram_gb) uniquely identifies instance_type for all rows.")
