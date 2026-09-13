"""
Decode the anonymized provider/family/vcpu label-encoding in the
multi-cloud-configuration dataset by recovering the implied hourly
price per configuration:

    cost = runtime_hours * hourly_price * nodes
    => hourly_price ~= cost / (runtime_hours * nodes)

We compute this per (provider, config) group (using only status == 'ok'
rows, since failed/timeout runs have unreliable runtime/cost pairs),
then rank configs within each provider by implied price. That ranking
is compared against known real-world price orderings to assign real
instance-type labels to the 0/1/2 codes.
"""
import pandas as pd

df = pd.read_csv("data/multi-cloud-configuration.csv", index_col=0)
ok = df[df["status"] == "ok"].copy()
ok["runtime_h"] = ok["target_runtime"] / 3600.0
ok["implied_price_per_node_hr"] = ok["target_cost"] / (ok["runtime_h"] * ok["nodes"])

for prov, cols in [
    ("A", ["A_family", "A_vcpu"]),
    ("B", ["B_family", "B_vcpu"]),
    ("C", ["C_family", "C_type", "C_vcpu"]),
]:
    sub = ok[ok["provider"] == prov]
    g = (
        sub.groupby(cols)["implied_price_per_node_hr"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean")
    )
    print(f"\n=== Provider {prov} ===")
    print(g.to_string())
