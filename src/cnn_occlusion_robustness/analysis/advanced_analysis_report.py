# advanced_analysis_report.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
import glob
from datetime import datetime
import argparse
from typing import Dict, Any

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class OcclusionRobustnessAnalyzer:
    """Advanced analyzer for occlusion robustness experimental results."""

    def __init__(self, results_dir: str, output_dir: str = "analysis_output"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.create_output_dirs()

        # Load all experimental data
        self.evaluation_data = self.load_evaluation_data()
        self.training_data = self.load_training_data()

        # Define effect categories for analysis
        self.effect_categories = {
            "clean": ["none"],
            "rain": ["light_rain", "moderate_rain", "heavy_rain"],
            "dust": ["light_dust", "moderate_dust", "heavy_dust"],
            "mixed": ["mixed_light", "mixed_heavy"],
        }

    def create_output_dirs(self):
        """Create organized output directory structure."""
        subdirs = ["figures", "tables", "reports", "statistical_tests"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def load_evaluation_data(self) -> pd.DataFrame:
        """Load and combine evaluation results from multiple experiments."""
        csv_files = glob.glob(os.path.join(self.results_dir, "evaluation_matrix_*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No evaluation matrix files found in {self.results_dir}"
            )

        # Load the most recent file or combine multiple runs
        latest_file = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(latest_file, index_col=0)

        print(f"Loaded evaluation data from: {latest_file}")
        print(f"Matrix shape: {df.shape}")

        return df

    def load_training_data(self) -> Dict:
        """Load training results data."""
        json_files = glob.glob(
            os.path.join(self.results_dir, "training_results_*.json")
        )

        if not json_files:
            print("Warning: No training results files found")
            return {}

        latest_file = max(json_files, key=os.path.getctime)
        with open(latest_file, "r") as f:
            data = json.load(f)

        print(f"Loaded training data from: {latest_file}")
        return data

    def generate_publication_figures(self):
        """Generate high-quality figures for publication."""
        print("Generating figures...")

        # Figure 1: Main performance heatmap
        self.create_publication_heatmap()

        # Figure 2: Robustness analysis
        self.create_robustness_comparison()

        # Figure 3: Category-wise performance
        self.create_category_analysis()

        # Figure 4: Statistical significance matrix
        self.create_significance_matrix()

        # Figure 5: Training dynamics
        self.create_training_dynamics_figure()

        print("Publication figures saved to figures/ directory")

    def create_publication_heatmap(self):
        """Create main performance heatmap for publication."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap with better formatting
        mask = self.evaluation_data.isna()
        sns.heatmap(
            self.evaluation_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Classification Accuracy"},
            mask=mask,
            ax=ax,
        )

        # --- This is the only block needed for formatting ---
        ax.set_title(
            "Cross-Domain Performance Matrix:\nTraining Condition vs. Test Condition",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Test Condition", fontsize=14, fontweight="bold")
        ax.set_ylabel("Training Condition", fontsize=14, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # --- End of formatting block ---

        # Get the actual labels for rows and columns
        row_labels = self.evaluation_data.index.to_list()
        col_labels = self.evaluation_data.columns.to_list()

        # Find the correct cell where the row label matches the column label
        for i, train_condition in enumerate(row_labels):
            try:
                # Find the column index 'j' that matches the current row's train_condition
                j = col_labels.index(train_condition)

                # Draw the rectangle at the correct (column, row) coordinate
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="blue", lw=3)
                )
            except ValueError:
                # This happens if a training condition is not in the test set.
                # We can safely ignore it.
                pass

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figures", "performance_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.output_dir, "figures", "performance_heatmap.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    def create_robustness_comparison(self):
        """Create comprehensive robustness comparison figure."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Robustness Analysis Across Training Conditions",
            fontsize=16,
            fontweight="bold",
        )

        # Calculate robustness metrics
        robustness_data = self.calculate_robustness_metrics()

        # 1. Clean vs Average Noisy Performance
        axes[0, 0].scatter(
            robustness_data["clean_accuracy"],
            robustness_data["mean_noisy_accuracy"],
            s=100,
            alpha=0.7,
            c=range(len(robustness_data)),
            cmap="viridis",
        )
        axes[0, 0].plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=2)
        axes[0, 0].set_xlabel("Clean Test Accuracy")
        axes[0, 0].set_ylabel("Mean Noisy Test Accuracy")
        axes[0, 0].set_title("Clean vs. Noisy Performance")
        axes[0, 0].grid(True, alpha=0.3)

        # Add model labels
        for i, model in enumerate(robustness_data.index):
            axes[0, 0].annotate(
                model.split("_")[0],
                (
                    robustness_data.iloc[i]["clean_accuracy"],
                    robustness_data.iloc[i]["mean_noisy_accuracy"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # 2. Robustness gap by training condition
        robustness_data["robustness_gap"].plot(
            kind="bar", ax=axes[0, 1], color="skyblue"
        )
        axes[0, 1].set_title("Robustness Gap (Clean - Mean Noisy)")
        axes[0, 1].set_ylabel("Accuracy Difference")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(axis="y", alpha=0.3)

        # 3. Worst-case performance
        robustness_data["worst_case_accuracy"].plot(
            kind="bar", ax=axes[0, 2], color="lightcoral"
        )
        axes[0, 2].set_title("Worst-Case Performance")
        axes[0, 2].set_ylabel("Minimum Accuracy")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].grid(axis="y", alpha=0.3)

        # 4. Consistency analysis (std of noisy performances)
        robustness_data["consistency"] = self.evaluation_data.apply(
            lambda row: np.std([row[col] for col in row.index if col != "none"]), axis=1
        )
        robustness_data["consistency"].plot(kind="bar", ax=axes[1, 0], color="gold")
        axes[1, 0].set_title("Performance Consistency\n(Lower = More Consistent)")
        axes[1, 0].set_ylabel("Standard Deviation")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(axis="y", alpha=0.3)

        # 5. Category-wise performance
        category_performance = self.calculate_category_performance()
        category_performance.plot(kind="bar", ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title("Performance by Effect Category")
        axes[1, 1].set_ylabel("Mean Accuracy")
        axes[1, 1].legend(
            title="Training Condition", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(axis="y", alpha=0.3)

        # 6. Training efficiency (best val acc vs final val acc)
        if self.training_data:
            training_efficiency = pd.DataFrame(
                {
                    model: {
                        "best_val_acc": data["best_val_acc"],
                        "final_val_acc": data["history"]["val_acc"][-1],
                    }
                    for model, data in self.training_data.items()
                }
            ).T

            axes[1, 2].scatter(
                training_efficiency["final_val_acc"],
                training_efficiency["best_val_acc"],
                s=100,
                alpha=0.7,
            )
            axes[1, 2].plot([0, 1], [0, 1], "r--", alpha=0.5)
            axes[1, 2].set_xlabel("Final Validation Accuracy")
            axes[1, 2].set_ylabel("Best Validation Accuracy")
            axes[1, 2].set_title("Training Convergence")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Training data\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figures", "robustness_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.output_dir, "figures", "robustness_analysis.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    def calculate_robustness_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive robustness metrics."""
        metrics = {}

        for model in self.evaluation_data.index:
            row = self.evaluation_data.loc[model]
            clean_acc = row["none"] if "none" in row.index else np.nan
            noisy_accs = [
                row[col] for col in row.index if col != "none" and not pd.isna(row[col])
            ]

            if noisy_accs:
                metrics[model] = {
                    "clean_accuracy": clean_acc,
                    "mean_noisy_accuracy": np.mean(noisy_accs),
                    "worst_case_accuracy": np.min(noisy_accs),
                    "best_noisy_accuracy": np.max(noisy_accs),
                    "robustness_gap": (
                        clean_acc - np.mean(noisy_accs)
                        if not pd.isna(clean_acc)
                        else np.nan
                    ),
                    "worst_case_gap": (
                        clean_acc - np.min(noisy_accs)
                        if not pd.isna(clean_acc)
                        else np.nan
                    ),
                    "relative_robustness": (
                        np.mean(noisy_accs) / clean_acc
                        if not pd.isna(clean_acc) and clean_acc > 0
                        else np.nan
                    ),
                }

        return pd.DataFrame(metrics).T

    def calculate_category_performance(self) -> pd.DataFrame:
        """Calculate performance by effect category."""
        category_data: dict[str, Any] = {}

        for model in self.evaluation_data.index:
            category_data[model] = {}
            row = self.evaluation_data.loc[model]

            for category, effects in self.effect_categories.items():
                available_effects = [eff for eff in effects if eff in row.index]
                if available_effects:
                    accs = [
                        row[eff] for eff in available_effects if not pd.isna(row[eff])
                    ]
                    category_data[model][category] = np.mean(accs) if accs else np.nan

        return pd.DataFrame(category_data)

    def create_category_analysis(self):
        """Create detailed category-wise analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Performance Analysis by Effect Category", fontsize=16, fontweight="bold"
        )

        category_perf = self.calculate_category_performance()

        # 1. Category comparison heatmap
        sns.heatmap(
            category_perf.T,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.5,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Performance by Category")
        axes[0, 0].set_xlabel("Training Condition")
        axes[0, 0].set_ylabel("Effect Category")

        # 2. In-domain vs cross-domain comparison
        in_domain_scores = []
        cross_domain_scores = []
        model_names = []

        for model in self.evaluation_data.index:
            train_effect = model.split("_")[0]
            row = self.evaluation_data.loc[model]

            # In-domain: same effect
            if train_effect in row.index:
                in_domain_scores.append(row[train_effect])
                model_names.append(model)

                # Cross-domain: different effects
                cross_effects = [
                    col
                    for col in row.index
                    if col != train_effect and not pd.isna(row[col])
                ]
                if cross_effects:
                    cross_domain_scores.append(
                        np.mean([row[col] for col in cross_effects])
                    )
                else:
                    cross_domain_scores.append(np.nan)

        axes[0, 1].scatter(in_domain_scores, cross_domain_scores, s=100, alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], "r--", alpha=0.5)
        axes[0, 1].set_xlabel("In-Domain Performance")
        axes[0, 1].set_ylabel("Cross-Domain Performance")
        axes[0, 1].set_title("In-Domain vs Cross-Domain")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Effect severity analysis
        effect_severity = self.calculate_effect_severity()
        effect_severity.plot(kind="bar", ax=axes[1, 0])
        axes[1, 0].set_title("Effect Severity (Performance Drop from Clean)")
        axes[1, 0].set_ylabel("Mean Accuracy Drop")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(axis="y", alpha=0.3)

        # 4. Training effect benefit
        training_benefit = self.calculate_training_benefit()
        training_benefit.plot(kind="bar", ax=axes[1, 1])
        axes[1, 1].set_title("Training Benefit (Specialized vs Clean Training)")
        axes[1, 1].set_ylabel("Accuracy Improvement")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figures", "category_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def calculate_effect_severity(self) -> pd.Series:
        """Calculate how much each effect degrades performance."""
        if "none" not in self.evaluation_data.index:
            return pd.Series()

        clean_performance = self.evaluation_data.loc["none"]
        severity = {}

        for effect in self.evaluation_data.columns:
            if effect != "none":
                severity[effect] = clean_performance["none"] - clean_performance[effect]

        return pd.Series(severity).sort_values(ascending=False)

    def calculate_training_benefit(self) -> pd.Series:
        """Calculate benefit of specialized training for each effect."""
        if "none" not in self.evaluation_data.index:
            return pd.Series()

        benefits = {}
        clean_model_row = self.evaluation_data.loc["none"]

        for effect in self.evaluation_data.columns:
            if effect != "none":
                # Find specialized model for this effect
                # A specialized model has the same name as the effect.
                specialized_models = [
                    model for model in self.evaluation_data.index if model == effect
                ]

                if specialized_models:
                    specialized_perf = self.evaluation_data.loc[
                        specialized_models[0], effect
                    ]
                    clean_model_perf = clean_model_row[effect]
                    benefits[effect] = specialized_perf - clean_model_perf

        return pd.Series(benefits).sort_values(ascending=False)

    def create_significance_matrix(self):
        """Create statistical significance comparison matrix."""
        print("Performing statistical significance tests...")

        models = list(self.evaluation_data.index)
        n_models = len(models)

        # Create p-value matrix
        p_values = np.ones((n_models, n_models))
        effect_sizes = np.zeros((n_models, n_models))

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # Get performance vectors for both models
                    perf1 = self.evaluation_data.loc[model1].dropna()
                    perf2 = self.evaluation_data.loc[model2].dropna()

                    # Find common test conditions
                    common_conditions = set(perf1.index) & set(perf2.index)

                    if len(common_conditions) > 1:
                        p1_common = [perf1[cond] for cond in common_conditions]
                        p2_common = [perf2[cond] for cond in common_conditions]

                        # Paired t-test
                        if len(p1_common) > 1:
                            statistic, p_val = stats.ttest_rel(p1_common, p2_common)
                            p_values[i, j] = p_val

                            # Calculate Cohen's d effect size
                            diff = np.array(p1_common) - np.array(p2_common)
                            effect_sizes[i, j] = (
                                np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                            )

        # Create significance matrix plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # P-value matrix
        mask = p_values == 1  # Mask diagonal
        sns.heatmap(
            p_values,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            xticklabels=models,
            yticklabels=models,
            mask=mask,
            ax=axes[0],
        )
        axes[0].set_title("Statistical Significance Matrix (p-values)")
        axes[0].set_xlabel("Model B")
        axes[0].set_ylabel("Model A")

        # Effect size matrix
        mask_diag = np.eye(n_models, dtype=bool)
        sns.heatmap(
            effect_sizes,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            xticklabels=models,
            yticklabels=models,
            mask=mask_diag,
            ax=axes[1],
        )
        axes[1].set_title("Effect Size Matrix (Cohen's d)")
        axes[1].set_xlabel("Model B")
        axes[1].set_ylabel("Model A (positive = A > B)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figures", "significance_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save statistical results
        significance_results = {
            "p_values": p_values.tolist(),
            "effect_sizes": effect_sizes.tolist(),
            "model_names": models,
            "interpretation": {
                "p_value_threshold": 0.05,
                "effect_size_interpretation": {
                    "small": 0.2,
                    "medium": 0.5,
                    "large": 0.8,
                },
            },
        }

        with open(
            os.path.join(
                self.output_dir, "statistical_tests", "significance_results.json"
            ),
            "w",
        ) as f:
            json.dump(significance_results, f, indent=2)

    def create_training_dynamics_figure(self):
        """Create training dynamics visualization."""
        if not self.training_data:
            print("No training data available for dynamics analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Dynamics Analysis", fontsize=16, fontweight="bold")

        # Collect all training curves
        all_train_acc = []
        all_val_acc = []
        all_train_loss = []
        all_val_loss = []
        labels = []

        for model_name, data in self.training_data.items():
            if "history" in data:
                history = data["history"]
                epochs = range(1, len(history["train_acc"]) + 1)

                axes[0, 0].plot(
                    epochs, history["train_acc"], label=model_name, alpha=0.8
                )
                axes[0, 1].plot(epochs, history["val_acc"], label=model_name, alpha=0.8)
                axes[1, 0].plot(
                    epochs, history["train_loss"], label=model_name, alpha=0.8
                )
                axes[1, 1].plot(
                    epochs, history["val_loss"], label=model_name, alpha=0.8
                )

                all_train_acc.extend(history["train_acc"])
                all_val_acc.extend(history["val_acc"])
                all_train_loss.extend(history["train_loss"])
                all_val_loss.extend(history["val_loss"])
                labels.extend([model_name] * len(history["train_acc"]))

        # Configure subplots
        axes[0, 0].set_title("Training Accuracy Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        axes[0, 1].set_title("Validation Accuracy Curves")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Training Loss Curves")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Validation Loss Curves")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figures", "training_dynamics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_comprehensive_report(self):
        """Generate a comprehensive markdown report with all findings."""
        print("Generating comprehensive analysis report...")

        report_path = os.path.join(
            self.output_dir, "reports", "comprehensive_analysis.md"
        )

        with open(report_path, "w") as f:
            # Header
            f.write("# Comprehensive Occlusion Robustness Analysis Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f)

            # Detailed Findings
            f.write("\n## Detailed Findings\n\n")
            self._write_detailed_findings(f)

            # Statistical Analysis
            f.write("\n## Statistical Analysis\n\n")
            self._write_statistical_analysis(f)

            # Recommendations
            f.write("\n## Recommendations\n\n")
            self._write_recommendations(f)

            # Appendix
            f.write("\n## Appendix\n\n")
            self._write_appendix(f)

        print(f"Comprehensive report saved to: {report_path}")

    def _write_executive_summary(self, f):
        """Write executive summary section."""
        robustness_metrics = self.calculate_robustness_metrics()

        f.write("### Key Findings\n\n")

        # Best performing model
        best_overall = robustness_metrics["mean_noisy_accuracy"].idxmax()
        best_score = robustness_metrics.loc[best_overall, "mean_noisy_accuracy"]

        f.write(
            f"- **Best Overall Model:** {best_overall} (Mean noisy accuracy: {best_score:.3f})\n"
        )

        # Most robust model
        most_robust = robustness_metrics["robustness_gap"].idxmin()
        robustness_gap = robustness_metrics.loc[most_robust, "robustness_gap"]

        f.write(
            f"- **Most Robust Model:** {most_robust} (Robustness gap: {robustness_gap:.3f})\n"
        )

        # Effect severity
        effect_severity = self.calculate_effect_severity()
        if not effect_severity.empty:
            worst_effect = effect_severity.idxmax()
            worst_drop = effect_severity.max()
            f.write(
                f"- **Most Challenging Effect:** {worst_effect} (Performance drop: {worst_drop:.3f})\n"
            )

        # Training benefit
        training_benefit = self.calculate_training_benefit()
        if not training_benefit.empty:
            best_benefit = training_benefit.idxmax()
            benefit_value = training_benefit.max()
            f.write(
                f"- **Highest Training Benefit:** {best_benefit} (Improvement: {benefit_value:.3f})\n"
            )

        f.write("\n### Performance Summary\n\n")
        f.write(f"- **Models Evaluated:** {len(self.evaluation_data.index)}\n")
        f.write(f"- **Test Conditions:** {len(self.evaluation_data.columns)}\n")
        f.write(
            f"- **Mean Clean Performance:** {robustness_metrics['clean_accuracy'].mean():.3f}\n"
        )
        f.write(
            f"- **Mean Noisy Performance:** {robustness_metrics['mean_noisy_accuracy'].mean():.3f}\n"
        )
        f.write(
            f"- **Average Robustness Gap:** {robustness_metrics['robustness_gap'].mean():.3f}\n"
        )

    def _write_detailed_findings(self, f):
        """Write detailed findings section."""
        f.write("### Performance Matrix Analysis\n\n")

        # In-domain vs cross-domain
        in_domain_perfs = []
        cross_domain_perfs = []

        for model in self.evaluation_data.index:
            train_effect = model.split("_")[0]
            row = self.evaluation_data.loc[model]

            if train_effect in row.index:
                in_domain_perfs.append(row[train_effect])
                cross_effects = [
                    col
                    for col in row.index
                    if col != train_effect and not pd.isna(row[col])
                ]
                if cross_effects:
                    cross_domain_perfs.append(
                        np.mean([row[col] for col in cross_effects])
                    )

        if in_domain_perfs and cross_domain_perfs:
            mean_in_domain = np.mean(in_domain_perfs)
            mean_cross_domain = np.mean(cross_domain_perfs)
            generalization_gap = mean_in_domain - mean_cross_domain

            f.write(f"- **In-domain Performance:** {mean_in_domain:.3f}\n")
            f.write(f"- **Cross-domain Performance:** {mean_cross_domain:.3f}\n")
            f.write(f"- **Generalization Gap:** {generalization_gap:.3f}\n\n")

        # Category analysis
        f.write("### Effect Category Analysis\n\n")
        category_perf = self.calculate_category_performance()

        for category in self.effect_categories.keys():
            if category in category_perf.columns:
                mean_perf = category_perf[category].mean()
                f.write(
                    f"- **{category.title()} Effects:** Mean accuracy {mean_perf:.3f}\n"
                )

        f.write("\n")

    def _write_statistical_analysis(self, f):
        """Write statistical analysis section."""
        f.write("### Statistical Significance Tests\n\n")
        f.write(
            "Paired t-tests were performed between all model pairs across common test conditions.\n\n"
        )

        # Load significance results if available
        sig_file = os.path.join(
            self.output_dir, "statistical_tests", "significance_results.json"
        )
        if os.path.exists(sig_file):
            with open(sig_file, "r") as sf:
                sig_results = json.load(sf)

            models = sig_results["model_names"]
            p_values = np.array(sig_results["p_values"])
            effect_sizes = np.array(sig_results["effect_sizes"])

            # Count significant differences
            significant_pairs = np.sum(p_values < 0.05) - len(
                models
            )  # Exclude diagonal
            total_pairs = len(models) * (len(models) - 1)

            f.write(
                f"- **Significant differences found:** {significant_pairs}/{total_pairs} model pairs (p < 0.05)\n"
            )

            # Large effect sizes
            large_effects = np.sum(np.abs(effect_sizes) > 0.8) - len(models)
            f.write(
                f"- **Large effect sizes:** {large_effects}/{total_pairs} comparisons (|d| > 0.8)\n\n"
            )

        f.write("*See significance_matrix.png for detailed pairwise comparisons.*\n\n")

    def _write_recommendations(self, f):
        """Write recommendations section."""
        robustness_metrics = self.calculate_robustness_metrics()

        f.write("### Model Selection Recommendations\n\n")

        # Best overall model
        best_overall = robustness_metrics["mean_noisy_accuracy"].idxmax()
        f.write(
            f"1. **For general deployment:** Use {best_overall} for best overall performance across conditions.\n"
        )

        # Most robust model
        most_robust = robustness_metrics["robustness_gap"].idxmin()
        f.write(
            f"2. **For unknown conditions:** Use {most_robust} for most consistent performance.\n"
        )

        # Clean performance model
        best_clean = robustness_metrics["clean_accuracy"].idxmax()
        f.write(
            f"3. **For clean conditions:** Use {best_clean} for optimal clean performance.\n\n"
        )

        f.write("### Training Recommendations\n\n")

        training_benefit = self.calculate_training_benefit()
        if not training_benefit.empty:
            beneficial_effects = training_benefit[training_benefit > 0.01]

            if len(beneficial_effects) > 0:
                f.write("**Beneficial specialized training for:**\n")
                for effect, benefit in beneficial_effects.items():
                    f.write(f"- {effect}: +{benefit:.3f} accuracy improvement\n")
            else:
                f.write(
                    "- Specialized training shows minimal benefit over clean training.\n"
                )

        f.write("\n### Deployment Considerations\n\n")
        f.write(
            "1. **Environment Assessment:** Evaluate expected occlusion types in deployment environment.\n"
        )
        f.write(
            "2. **Performance Monitoring:** Implement performance monitoring to detect distribution shifts.\n"
        )
        f.write(
            "3. **Ensemble Methods:** Consider ensemble approaches for critical applications.\n"
        )
        f.write(
            "4. **Data Augmentation:** Use appropriate augmentation during training for robustness.\n"
        )

    def _write_appendix(self, f):
        """Write appendix with technical details."""
        f.write("### Experimental Setup\n\n")
        f.write("**Effect Definitions:**\n")
        for effect_name, effect_list in self.effect_categories.items():
            f.write(f"- {effect_name}: {', '.join(effect_list)}\n")

        f.write("\n**Metrics Definitions:**\n")
        f.write("- **Robustness Gap:** Clean accuracy - Mean noisy accuracy\n")
        f.write(
            "- **Consistency:** Standard deviation of performance across noisy conditions\n"
        )
        f.write(
            "- **Training Benefit:** Specialized model accuracy - Clean model accuracy on same condition\n"
        )

        f.write("\n**Files Generated:**\n")
        f.write("- `figures/`: All visualization files (PNG and PDF formats)\n")
        f.write("- `tables/`: Numerical results in CSV format\n")
        f.write("- `statistical_tests/`: Statistical analysis results\n")
        f.write("- `reports/`: This comprehensive report\n")

    def save_summary_tables(self):
        """Save all numerical results as CSV files."""
        print("Saving summary tables...")

        # Robustness metrics
        robustness_metrics = self.calculate_robustness_metrics()
        robustness_metrics.to_csv(
            os.path.join(self.output_dir, "tables", "robustness_metrics.csv")
        )

        # Category performance
        category_perf = self.calculate_category_performance()
        category_perf.to_csv(
            os.path.join(self.output_dir, "tables", "category_performance.csv")
        )

        # Effect severity
        effect_severity = self.calculate_effect_severity()
        effect_severity.to_csv(
            os.path.join(self.output_dir, "tables", "effect_severity.csv")
        )

        # Training benefit
        training_benefit = self.calculate_training_benefit()
        training_benefit.to_csv(
            os.path.join(self.output_dir, "tables", "training_benefit.csv")
        )

        print("Summary tables saved to tables/ directory")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive analysis pipeline...")

        # Generate all figures
        self.generate_publication_figures()

        # Save numerical results
        self.save_summary_tables()

        # Generate comprehensive report
        self.generate_comprehensive_report()

        print(f"\nAnalysis complete! All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("📊 figures/performance_heatmap.png - Main performance matrix")
        print("📈 figures/robustness_analysis.png - Comprehensive robustness analysis")
        print("📋 figures/category_analysis.png - Effect category analysis")
        print("🔬 figures/significance_matrix.png - Statistical significance tests")
        print("📚 reports/comprehensive_analysis.md - Detailed analysis report")
        print("📄 tables/ - All numerical results in CSV format")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Analysis of Occlusion Robustness Results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experimental results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = OcclusionRobustnessAnalyzer(args.results_dir, args.output_dir)

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
