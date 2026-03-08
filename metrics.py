# -*- coding: utf-8 -*-


class Metrics:
    @staticmethod
    def compute(df_baseline, df_strategy):
        accuracy = df_strategy["correct"].mean()
        avg_tokens = df_strategy["tokens"].mean()
        avg_time = df_strategy["time"].mean()

        compression_ratio = avg_tokens / df_baseline["tokens"].mean()
        speedup_ratio = df_baseline["time"].mean() / avg_time

        metrics = {
            "accuracy": float(accuracy),
            "avg_tokens": float(avg_tokens),
            "avg_time": float(avg_time),
            "compression_ratio": float(compression_ratio),
            "speedup_ratio": float(speedup_ratio),
        }

        if "outcome_eff" in df_strategy.columns and "outcome_eff" in df_baseline.columns:
            outcome_eff_mean = df_strategy["outcome_eff"].mean()
            baseline_oe = df_baseline["outcome_eff"].mean()
            metrics["outcome_eff_mean"] = float(outcome_eff_mean)
            if baseline_oe and baseline_oe != 0:
                metrics["outcome_eff_ratio_vs_baseline"] = float(outcome_eff_mean / baseline_oe)

        return metrics
