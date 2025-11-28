#!/usr/bin/env python3
"""
Compute benchmark metrics and generate reports.
Calculates:
- % of code with compilation errors
- % of code with semantic errors
- % of code with security issues
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


class MetricsComputer:
    """Compute and visualize benchmark metrics."""

    def __init__(self, results_dir: str = "./benchmark/results"):
        self.results_dir = Path(results_dir)
        self.summary_dir = self.results_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def load_analysis_results(self) -> List[Dict]:
        """Load all analysis results."""
        analysis_path = self.summary_dir / "analysis_results.json"

        if not analysis_path.exists():
            print(f"Analysis results not found at {analysis_path}")
            print("Run analyze_multi.py first")
            return []

        with open(analysis_path, 'r') as f:
            return json.load(f)

    def compute_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        Compute metrics per model.

        Metrics:
        - Compilation Error %: Programs that failed to compile/parse
        - Semantic Error %: Programs that compiled but failed tests
          (wrong output, timeout, crash, memory error, no output)
        - Security Issue %: Programs with security vulnerabilities

        Returns:
            DataFrame with metrics for each model
        """
        # Group results by model
        model_stats = defaultdict(lambda: {
            'total': 0,
            'compilation_errors': 0,
            'semantic_errors': 0,
            'security_issues': 0,
            'successful_compilations': 0,
            'tests_run': 0,
            'tests_passed': 0,
            'security_analyzed': 0,
            # Failure breakdown for semantic errors
            'failure_breakdown': defaultdict(int),
            # Language breakdown
            'by_language': defaultdict(lambda: {
                'total': 0,
                'compilation_errors': 0,
                'semantic_errors': 0,
                'security_issues': 0
            })
        })

        for result in results:
            model = result['model']
            stats = model_stats[model]

            stats['total'] += 1

            # Language tracking
            lang = result.get('detected_language', 'unknown')
            lang_stats = stats['by_language'][lang]
            lang_stats['total'] += 1

            # Compilation status
            if result.get('compilation'):
                if not result['compilation']['compiles']:
                    stats['compilation_errors'] += 1
                    lang_stats['compilation_errors'] += 1
                else:
                    stats['successful_compilations'] += 1

            # Semantic errors = test case failures (NEW definition)
            # This includes: wrong output, timeout, crash, memory error, no output
            test_exec = result.get('test_execution', {})
            if test_exec and not test_exec.get('skipped'):
                stats['tests_run'] += test_exec.get('total_tests', 0)
                stats['tests_passed'] += test_exec.get('passed', 0)

                # A program has semantic errors if ANY test failed
                if result.get('has_semantic_error') or test_exec.get('failed', 0) > 0:
                    stats['semantic_errors'] += 1
                    lang_stats['semantic_errors'] += 1

                # Track failure types
                failure_breakdown = result.get('failure_breakdown', {})
                for failure_type, count in failure_breakdown.items():
                    stats['failure_breakdown'][failure_type] += count

            # Security issues (CodeQL or Bandit)
            security = result.get('security', result.get('codeql', {}))
            if security and security.get('success'):
                stats['security_analyzed'] += 1

                security_issues = security.get('security_issues', [])
                if security_issues:
                    stats['security_issues'] += 1
                    lang_stats['security_issues'] += 1

        # Convert to DataFrame with percentages
        metrics = []
        for model, stats in model_stats.items():
            total = stats['total']
            if total == 0:
                continue

            # Semantic error % is calculated over compilable samples
            compilable = stats['successful_compilations']

            metrics.append({
                'Model': model,
                'Total Samples': total,
                'Compilation Error %': (stats['compilation_errors'] / total) * 100,
                'Semantic Error %': (stats['semantic_errors'] / compilable * 100) if compilable > 0 else 0,
                'Security Issue %': (stats['security_issues'] / stats['security_analyzed'] * 100) if stats['security_analyzed'] > 0 else 0,
                'Successful Compilations': stats['successful_compilations'],
                'Tests Run': stats['tests_run'],
                'Tests Passed': stats['tests_passed'],
                'Security Analyzed': stats['security_analyzed'],
                'Failure Breakdown': dict(stats['failure_breakdown'])
            })

        df = pd.DataFrame(metrics)
        return df.sort_values('Model') if not df.empty else df

    def compute_detailed_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        Compute detailed metrics broken down by dataset.

        Returns:
            DataFrame with metrics for each model-dataset combination
        """
        # Group by model and dataset
        stats = defaultdict(lambda: {
            'total': 0,
            'compilation_errors': 0,
            'semantic_errors': 0,
            'security_issues': 0,
            'successful_compilations': 0,
            'security_analyzed': 0,
            'failure_breakdown': defaultdict(int)
        })

        for result in results:
            key = (result['model'], result['dataset'])
            s = stats[key]

            s['total'] += 1

            # Compilation
            if result.get('compilation'):
                if not result['compilation']['compiles']:
                    s['compilation_errors'] += 1
                else:
                    s['successful_compilations'] += 1

            # Semantic errors = test failures
            test_exec = result.get('test_execution', {})
            if test_exec and not test_exec.get('skipped'):
                if result.get('has_semantic_error') or test_exec.get('failed', 0) > 0:
                    s['semantic_errors'] += 1

                # Track failure breakdown
                for ft, count in result.get('failure_breakdown', {}).items():
                    s['failure_breakdown'][ft] += count

            # Security issues
            security = result.get('security', result.get('codeql', {}))
            if security and security.get('success'):
                s['security_analyzed'] += 1
                if security.get('security_issues'):
                    s['security_issues'] += 1

        # Convert to DataFrame
        detailed = []
        for (model, dataset), s in stats.items():
            total = s['total']
            if total == 0:
                continue

            compilable = s['successful_compilations']

            detailed.append({
                'Model': model,
                'Dataset': dataset,
                'Total': total,
                'Compilation Error %': (s['compilation_errors'] / total) * 100,
                'Semantic Error %': (s['semantic_errors'] / compilable * 100) if compilable > 0 else 0,
                'Security Issue %': (s['security_issues'] / s['security_analyzed'] * 100) if s['security_analyzed'] > 0 else 0,
                'Failure Breakdown': dict(s['failure_breakdown'])
            })

        df = pd.DataFrame(detailed)
        return df.sort_values(['Model', 'Dataset']) if not df.empty else df

    def compute_failure_breakdown(self, results: List[Dict]) -> pd.DataFrame:
        """
        Compute detailed failure breakdown by type.

        Returns:
            DataFrame with counts of each failure type per model
        """
        breakdown = defaultdict(lambda: defaultdict(int))

        for result in results:
            model = result['model']
            for failure_type, count in result.get('failure_breakdown', {}).items():
                breakdown[model][failure_type] += count

        rows = []
        for model, failures in breakdown.items():
            row = {'Model': model}
            total = sum(failures.values())
            for ft, count in failures.items():
                row[ft] = count
                row[f'{ft} %'] = (count / total * 100) if total > 0 else 0
            row['Total Failures'] = total
            rows.append(row)

        return pd.DataFrame(rows).sort_values('Model') if rows else pd.DataFrame()

    def generate_visualizations(self, metrics_df: pd.DataFrame):
        """Generate comparison visualizations."""
        viz_dir = self.summary_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

        # 1. Comparison bar chart
        fig, ax = plt.subplots(figsize=(14, 6))

        x = range(len(metrics_df))
        width = 0.25

        ax.bar([i - width for i in x], metrics_df['Compilation Error %'],
               width, label='Compilation Errors', color='#e74c3c')
        ax.bar(x, metrics_df['Semantic Error %'],
               width, label='Semantic Errors', color='#f39c12')
        ax.bar([i + width for i in x], metrics_df['Security Issue %'],
               width, label='Security Issues', color='#c0392b')

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Code Quality Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Heatmap
        heatmap_data = metrics_df.set_index('Model')[
            ['Compilation Error %', 'Semantic Error %', 'Security Issue %']
        ]

        fig, ax = plt.subplots(figsize=(10, len(metrics_df) * 0.8))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Percentage (%)'}, ax=ax)
        ax.set_title('Error Rate Heatmap by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig(viz_dir / "error_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Success rate pie charts (optional)
        for idx, row in metrics_df.iterrows():
            fig, ax = plt.subplots(figsize=(8, 8))

            compilation_success = 100 - row['Compilation Error %']
            sizes = [compilation_success, row['Compilation Error %']]
            labels = ['Success', 'Compilation Error']
            colors = ['#2ecc71', '#e74c3c']

            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 12})
            ax.set_title(f"{row['Model']} - Compilation Success Rate",
                        fontsize=14, fontweight='bold')

            plt.savefig(viz_dir / f"{row['Model']}_success_rate.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\nVisualizations saved to: {viz_dir}")

    def generate_report(self, metrics_df: pd.DataFrame, detailed_df: pd.DataFrame, failure_df: pd.DataFrame = None):
        """Generate markdown report."""
        report_path = self.summary_dir / "benchmark_report.md"

        with open(report_path, 'w') as f:
            f.write("# Multi-LLM Code Generation Benchmark Report\n\n")

            # Metrics definitions
            f.write("## Metrics Definitions\n\n")
            f.write("- **Compilation Error %**: Programs that failed to compile/parse\n")
            f.write("- **Semantic Error %**: Programs that compiled but failed test cases\n")
            f.write("  - Includes: wrong output, timeout/hang, runtime crash, memory error, no output\n")
            f.write("- **Security Issue %**: Programs with security vulnerabilities (CodeQL/Bandit)\n\n")

            # Overall summary
            f.write("## Overall Metrics\n\n")
            f.write("| Model | Total Samples | Compilation Error % | Semantic Error % | Security Issue % |\n")
            f.write("|-------|---------------|---------------------|------------------|------------------|\n")

            for _, row in metrics_df.iterrows():
                f.write(f"| {row['Model']} | {row['Total Samples']} | "
                       f"{row['Compilation Error %']:.1f}% | "
                       f"{row['Semantic Error %']:.1f}% | "
                       f"{row['Security Issue %']:.1f}% |\n")

            f.write("\n")

            # Best performers
            if len(metrics_df) > 0:
                f.write("## Key Findings\n\n")

                best_compilation = metrics_df.loc[metrics_df['Compilation Error %'].idxmin()]
                f.write(f"- **Best Compilation Rate**: {best_compilation['Model']} "
                       f"({100 - best_compilation['Compilation Error %']:.1f}% success)\n")

                best_semantic = metrics_df.loc[metrics_df['Semantic Error %'].idxmin()]
                f.write(f"- **Fewest Semantic Errors**: {best_semantic['Model']} "
                       f"({best_semantic['Semantic Error %']:.1f}%)\n")

                best_security = metrics_df.loc[metrics_df['Security Issue %'].idxmin()]
                f.write(f"- **Fewest Security Issues**: {best_security['Model']} "
                       f"({best_security['Security Issue %']:.1f}%)\n")

                f.write("\n")

            # Semantic Error Breakdown
            if failure_df is not None and not failure_df.empty:
                f.write("## Semantic Error Breakdown\n\n")
                f.write("Breakdown of failure types (all count as 'failed to solve task'):\n\n")
                f.write("| Model | Wrong Output | Timeout | Crash | Memory Error | No Output | Total |\n")
                f.write("|-------|--------------|---------|-------|--------------|-----------|-------|\n")

                for _, row in failure_df.iterrows():
                    f.write(f"| {row['Model']} | "
                           f"{row.get('wrong_output', 0)} | "
                           f"{row.get('timeout', 0)} | "
                           f"{row.get('crash', 0)} | "
                           f"{row.get('memory_error', 0)} | "
                           f"{row.get('no_output', 0)} | "
                           f"{row.get('Total Failures', 0)} |\n")

                f.write("\n")

            # Detailed breakdown
            f.write("## Detailed Breakdown by Dataset\n\n")
            f.write("| Model | Dataset | Total | Compilation Error % | Semantic Error % | Security Issue % |\n")
            f.write("|-------|---------|-------|---------------------|------------------|------------------|\n")

            for _, row in detailed_df.iterrows():
                f.write(f"| {row['Model']} | {row['Dataset']} | {row['Total']} | "
                       f"{row['Compilation Error %']:.1f}% | "
                       f"{row['Semantic Error %']:.1f}% | "
                       f"{row['Security Issue %']:.1f}% |\n")

            f.write("\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("![Model Comparison](visualizations/model_comparison.png)\n\n")
            f.write("![Error Heatmap](visualizations/error_heatmap.png)\n\n")

        print(f"\nReport generated: {report_path}")

    def run(self):
        """Run complete metrics computation and reporting."""
        print("Loading analysis results...")
        results = self.load_analysis_results()

        if not results:
            print("No results to process")
            return

        print(f"Processing {len(results)} analysis results...")

        # Compute metrics
        metrics_df = self.compute_metrics(results)
        detailed_df = self.compute_detailed_metrics(results)
        failure_df = self.compute_failure_breakdown(results)

        # Save to CSV
        metrics_csv = self.summary_dir / "metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to: {metrics_csv}")

        detailed_csv = self.summary_dir / "detailed_metrics.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"Detailed metrics saved to: {detailed_csv}")

        if not failure_df.empty:
            failure_csv = self.summary_dir / "failure_breakdown.csv"
            failure_df.to_csv(failure_csv, index=False)
            print(f"Failure breakdown saved to: {failure_csv}")

        # Generate visualizations
        print("\nGenerating visualizations...")
        if not metrics_df.empty:
            self.generate_visualizations(metrics_df)

        # Generate report
        print("Generating report...")
        self.generate_report(metrics_df, detailed_df, failure_df)

        # Print summary to console
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        if not metrics_df.empty:
            # Select columns to display (exclude complex types like dict)
            display_cols = ['Model', 'Total Samples', 'Compilation Error %', 'Semantic Error %', 'Security Issue %']
            display_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]]
            print(display_df.to_string(index=False))
        else:
            print("No metrics computed")
        print("="*60)

        if not failure_df.empty:
            print("\nSEMANTIC ERROR BREAKDOWN")
            print("="*60)
            print(failure_df.to_string(index=False))
            print("="*60)


if __name__ == "__main__":
    computer = MetricsComputer()
    computer.run()
