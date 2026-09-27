import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import bytes_moved, energy, latency, memory


def to_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def fit_latency(df):
    d = df[(~df["is_validation"]) & (~df["oom"]) & df["latency_s"].notna()]

    S = d["S"].to_numpy()
    B = d["B"].to_numpy()
    y = d["latency_s"].to_numpy()

    def unpack(z):
        return {
            "t0": float(np.exp(z[0])),
            "compute_flops_s": float(np.exp(z[1])),
            "bandwidth_bytes_s": float(np.exp(z[2])),
        }

    def residual(z):
        pred = latency(S, B, unpack(z))
        return np.log(pred) - np.log(y)

    x0 = np.log([1e-4, 4e12, 1e11])
    lo = np.log([1e-7, 1e9, 1e8])
    hi = np.log([1e-1, 1e14, 1e13])

    result = least_squares(residual, x0, bounds=(lo, hi))
    return unpack(result.x)


def fit_energy(df):
    d = df[(~df["is_validation"]) & (~df["oom"]) & df["energy_j"].notna()]

    S = d["S"].to_numpy()
    B = d["B"].to_numpy()
    y = d["energy_j"].to_numpy()

    F = d["flops_pred"].to_numpy() / 1e9
    Q = bytes_moved(S, B) / 1e9

    def unpack(z):
        return {
            "e0": float(np.exp(z[0])),
            "e_per_gflop": float(np.exp(z[1])),
            "e_per_gbyte": float(np.exp(z[2])),
        }

    def residual(z):
        p = unpack(z)
        pred = p["e0"] + p["e_per_gflop"] * F + p["e_per_gbyte"] * Q
        return np.log(pred) - np.log(y)

    x0 = np.log([0.02, 0.01, 0.1])
    lo = np.log([1e-8, 1e-12, 1e-12])
    hi = np.log([100, 100, 100])

    result = least_squares(residual, x0, bounds=(lo, hi))
    return unpack(result.x)


def mape(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return float(np.mean(np.abs((y - pred) / y)) * 100)


def r2(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def print_metrics(df, name, measured, predicted):
    for split_name, mask in [
        ("calibration", ~df["is_validation"]),
        ("validation", df["is_validation"]),
    ]:
        d = df[mask & (~df["oom"]) & df[measured].notna() & df[predicted].notna()]
        if len(d):
            print(
                f"{name} {split_name}: "
                f"MAPE={mape(d[measured], d[predicted]):.2f}% "
                f"R2={r2(d[measured], d[predicted]):.4f}"
            )


def plot_surface(df, pred_fn, measured_col, scale, zlabel, path):
    s = np.array(sorted(df["S"].unique()))
    b = np.array(sorted(df["B"].unique()))
    S, B = np.meshgrid(s, b)
    Z = pred_fn(S, B) / scale

    train = df[
        (~df["is_validation"])
        & (~df["oom"])
        & df[measured_col].notna()
    ]

    val = df[
        df["is_validation"]
        & (~df["oom"])
        & df[measured_col].notna()
    ]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(S, B, Z, alpha=0.35)

    ax.scatter(
        train["S"],
        train["B"],
        train[measured_col] / scale,
        s=22,
        label="Calibration",
    )

    ax.scatter(
        val["S"],
        val["B"],
        val[measured_col] / scale,
        s=30,
        marker="^",
        label="Validation",
    )

    ax.set_xlabel("Image size S")
    ax.set_ylabel("Batch size B")
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_parity(df, measured_col, predicted_col, scale, label, path):
    train = df[
        (~df["is_validation"])
        & (~df["oom"])
        & df[measured_col].notna()
        & df[predicted_col].notna()
    ]

    val = df[
        df["is_validation"]
        & (~df["oom"])
        & df[measured_col].notna()
        & df[predicted_col].notna()
    ]

    x_train = train[measured_col].to_numpy() / scale
    y_train = train[predicted_col].to_numpy() / scale
    x_val = val[measured_col].to_numpy() / scale
    y_val = val[predicted_col].to_numpy() / scale

    values = np.concatenate([x_train, y_train, x_val, y_val])
    values = values[np.isfinite(values) & (values > 0)]

    lo = values.min()
    hi = values.max()

    plt.figure(figsize=(6.5, 6.5))

    plt.scatter(
        x_train,
        y_train,
        s=28,
        alpha=0.75,
        label="Calibration",
    )

    plt.scatter(
        x_val,
        y_val,
        s=34,
        alpha=0.85,
        marker="^",
        label="Validation",
    )

    plt.plot([lo, hi], [lo, hi], "--", label="Ideal")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(lo * 0.85, hi * 1.15)
    plt.ylim(lo * 0.85, hi * 1.15)
    plt.xlabel(f"Measured {label}")
    plt.ylabel(f"Predicted {label}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_relative_error(df, measured_col, predicted_col, scale, label, path):
    d = df[
        (~df["oom"])
        & df[measured_col].notna()
        & df[predicted_col].notna()
    ].copy()

    d["relative_error"] = (
        (d[predicted_col] - d[measured_col])
        / d[measured_col]
        * 100
    )

    train = d[~d["is_validation"]]
    val = d[d["is_validation"]]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        train[measured_col] / scale,
        train["relative_error"],
        s=25,
        alpha=0.7,
        label="Calibration",
    )

    plt.scatter(
        val[measured_col] / scale,
        val["relative_error"],
        s=30,
        alpha=0.8,
        marker="^",
        label="Validation",
    )

    plt.axhline(0, linestyle="--")
    plt.xscale("log")
    plt.xlabel(f"Measured {label}")
    plt.ylabel("Relative error [%]")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_oom(df, path):
    ok = df[~df["oom"]]
    oom = df[df["oom"]]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        ok["S"],
        ok["B"],
        s=32,
        alpha=0.7,
        label="Fits in memory",
    )

    if len(oom):
        plt.scatter(
            oom["S"],
            oom["B"],
            s=70,
            marker="x",
            label="OOM",
        )

    plt.yscale("log", base=2)
    plt.xlabel("Image size S")
    plt.ylabel("Batch size B")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv("results/measurements.csv")

    df["is_validation"] = to_bool(df["is_validation"])
    df["oom"] = to_bool(df["oom"])

    theta = fit_latency(df)
    theta_energy = fit_energy(df)

    df["memory_pred_bytes"] = memory(df["S"], df["B"])
    df["latency_pred_s"] = latency(df["S"], df["B"], theta)
    df["energy_pred_j"] = energy(df["S"], df["B"], theta_energy)

    print("latency theta:")
    print(json.dumps(theta, indent=2))

    print("energy theta:")
    print(json.dumps(theta_energy, indent=2))

    print_metrics(df, "memory", "memory_bytes", "memory_pred_bytes")
    print_metrics(df, "latency", "latency_s", "latency_pred_s")
    print_metrics(df, "energy", "energy_j", "energy_pred_j")

    os.makedirs("results/figures", exist_ok=True)

    with open("results/theta.json", "w") as f:
        json.dump(
            {
                "latency": theta,
                "energy": theta_energy,
            },
            f,
            indent=2,
        )

    df.to_csv("results/measurements.csv", index=False)

    plot_surface(
        df,
        memory,
        "memory_bytes",
        2**20,
        "Memory [MiB]",
        "results/figures/memory_surface.png",
    )

    plot_surface(
        df,
        lambda S, B: latency(S, B, theta),
        "latency_s",
        1e-3,
        "Latency [ms]",
        "results/figures/latency_surface.png",
    )

    plot_surface(
        df,
        lambda S, B: energy(S, B, theta_energy),
        "energy_j",
        1,
        "Energy [J]",
        "results/figures/energy_surface.png",
    )

    plot_parity(
        df,
        "memory_bytes",
        "memory_pred_bytes",
        2**20,
        "memory [MiB]",
        "results/figures/memory_parity.png",
    )

    plot_parity(
        df,
        "latency_s",
        "latency_pred_s",
        1e-3,
        "latency [ms]",
        "results/figures/latency_parity.png",
    )

    plot_parity(
        df,
        "energy_j",
        "energy_pred_j",
        1,
        "energy [J]",
        "results/figures/energy_parity.png",
    )

    plot_relative_error(
        df,
        "memory_bytes",
        "memory_pred_bytes",
        2**20,
        "memory [MiB]",
        "results/figures/memory_error.png",
    )

    plot_relative_error(
        df,
        "latency_s",
        "latency_pred_s",
        1e-3,
        "latency [ms]",
        "results/figures/latency_error.png",
    )

    plot_relative_error(
        df,
        "energy_j",
        "energy_pred_j",
        1,
        "energy [J]",
        "results/figures/energy_error.png",
    )

    plot_oom(
        df,
        "results/figures/oom_grid.png",
    )


if __name__ == "__main__":
    main()
