
import pandas as pd

def prepare_data(df):
    threshold = df["model_score"].quantile(0.7)
    df["approved"] = (df["model_score"] > threshold).astype(int)

    threshold_m = df["mitigated_score"].quantile(0.7)
    df["approved_mitigated"] = (df["mitigated_score"] > threshold_m).astype(int)

    return df

def compute_fairness(df, target, sensitive):
    groups = df[sensitive].unique()
    rates = {g: df[df[sensitive]==g][target].mean() for g in groups}

    ref = list(rates.keys())[0]
    ref_rate = rates[ref]

    result = []
    for g, rate in rates.items():
        result.append({
            "group": g,
            "selection_rate": round(rate,3),
            "SPD": round(rate - ref_rate,3),
            "DIR": round(rate / ref_rate if ref_rate else 0,3)
        })

    return pd.DataFrame(result)

def threshold_analysis(df, score_col, sensitive):
    results = []
    for q in [0.5, 0.6, 0.7]:
        threshold = df[score_col].quantile(q)
        df["temp"] = (df[score_col] > threshold).astype(int)

        fairness = compute_fairness(df, "temp", sensitive)

        results.append({
            "threshold": q,
            "fairness": fairness,
            "approval_rate": df["temp"].mean()
        })

    return results
