import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path

st.set_page_config(page_title="RESEARCH - Continual Learning Benchmark", layout="wide")

METHOD_DESCRIPTIONS = {
    "Naive": "A baseline method that trains on each task sequentially without any continual learning technique. Simply fine-tunes the model on each new task. Serves as a lower bound for comparison.",
    "EWC": "Elastic Weight Consolidation (EWC) adds a regularization term to the loss that penalizes changes to important parameters (as measured by Fisher information). Helps prevent catastrophic forgetting by protecting important weights.",
    "GEM": "Gradient Episodic Memory (GEM) stores a small buffer of examples from previous tasks and constrains gradients to prevent interference with previous tasks. Uses quadratic programming to project gradients.",
    "Replay": "Experience Replay stores a subset of examples from previous tasks and replays them during training on new tasks. Simple but effective approach to prevent forgetting.",
    "ResearchHybrid": "A novel method combining EWC regularization on backbone weights, selective replay using highest softmax entropy samples (hardest examples), and dynamic task heads (separate output head per task)."
}

METHOD_HYPERPARAMS = {
    "Naive": {
        "lr": 0.001,
        "epochs": 10,
        "optimizer": "Adam"
    },
    "EWC": {
        "lr": 0.001,
        "epochs": 10,
        "ewc_lambda": 5000,
        "fisher_computation": "online"
    },
    "GEM": {
        "lr": 0.001,
        "epochs": 10,
        "memory_size": 200,
        "constraint_projection": "quadratic"
    },
    "Replay": {
        "lr": 0.001,
        "epochs": 10,
        "memory_size": 100,
        "replay_ratio": 0.5
    },
    "ResearchHybrid": {
        "lr": 0.001,
        "epochs": 10,
        "ewc_lambda": 1000,
        "replay_size": 300,
        "heads_lr": 0.01,
        "replay_selection": "entropy"
    }
}

def load_results(results_dir: Path):
    results = {}
    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            with open(json_file) as f:
                results[json_file.stem] = json.load(f)
    return results

def compute_metrics(results_matrix, method_name):
    import numpy as np
    
    num_tasks = results_matrix.shape[0]
    
    avg_accuracy = float(np.mean(np.diag(results_matrix)))
    
    off_diagonal = []
    for i in range(num_tasks):
        for j in range(i):
            off_diagonal.append(results_matrix[i][j] - results_matrix[i-1][j])
    backward_transfer = float(np.mean(off_diagonal)) if off_diagonal else 0.0
    
    forward_scores = []
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            forward_scores.append(results_matrix[i][j])
    forward_transfer = float(np.mean(forward_scores)) if forward_scores else 0.0
    
    forgetting = []
    for j in range(num_tasks):
        best_after_training = results_matrix[j][j]
        drops = []
        for i in range(j+1, num_tasks):
            drop = best_after_training - results_matrix[i][j]
            drops.append(drop)
        if drops:
            forgetting.append(max(drops))
    forgetting = float(np.mean(forgetting)) if forgetting else 0.0
    
    return {
        "Method": method_name,
        "Avg Accuracy": round(avg_accuracy, 4),
        "Backward Transfer": round(backward_transfer, 4),
        "Forward Transfer": round(forward_transfer, 4),
        "Forgetting": round(forgetting, 4)
    }

def create_dummy_results(run_name):
    import numpy as np
    
    methods = ["Naive", "EWC", "GEM", "Replay", "ResearchHybrid"]
    num_tasks = 5
    
    results = {
        "run_name": run_name,
        "dataset": "MNIST",
        "methods": {}
    }
    
    base_accuracies = {
        "Naive": np.array([
            [0.95, 0.72, 0.58, 0.45, 0.38],
            [0.68, 0.88, 0.65, 0.52, 0.41],
            [0.55, 0.62, 0.82, 0.58, 0.48],
            [0.42, 0.48, 0.55, 0.75, 0.52],
            [0.35, 0.40, 0.45, 0.48, 0.70]
        ]),
        "EWC": np.array([
            [0.94, 0.78, 0.68, 0.58, 0.50],
            [0.75, 0.90, 0.72, 0.62, 0.54],
            [0.62, 0.68, 0.85, 0.68, 0.58],
            [0.52, 0.58, 0.62, 0.80, 0.62],
            [0.45, 0.50, 0.55, 0.58, 0.75]
        ]),
        "GEM": np.array([
            [0.93, 0.80, 0.72, 0.65, 0.58],
            [0.78, 0.91, 0.75, 0.68, 0.60],
            [0.68, 0.72, 0.86, 0.72, 0.64],
            [0.58, 0.62, 0.68, 0.82, 0.68],
            [0.52, 0.56, 0.60, 0.64, 0.78]
        ]),
        "Replay": np.array([
            [0.94, 0.82, 0.74, 0.66, 0.58],
            [0.80, 0.92, 0.78, 0.70, 0.62],
            [0.70, 0.74, 0.88, 0.74, 0.66],
            [0.60, 0.64, 0.70, 0.84, 0.70],
            [0.54, 0.58, 0.62, 0.66, 0.80]
        ]),
        "ResearchHybrid": np.array([
            [0.96, 0.85, 0.78, 0.70, 0.62],
            [0.82, 0.93, 0.80, 0.72, 0.64],
            [0.72, 0.76, 0.90, 0.76, 0.68],
            [0.62, 0.68, 0.72, 0.86, 0.72],
            [0.56, 0.60, 0.64, 0.68, 0.82]
        ])
    }
    
    for method in methods:
        results["methods"][method] = {
            "results_matrix": base_accuracies[method].tolist(),
            "training_curves": {}
        }
        
        for task_id in range(num_tasks):
            results["methods"][method]["training_curves"][task_id] = {
                "train_acc": [0.5 + i * 0.1 for i in range(task_id + 1)],
                "val_acc": [base_accuracies[method][task_id][t] for t in range(task_id + 1)]
            }
    
    return results

def highlight_best_worst(df, column):
    if column in ["Method"]:
        return [""] * len(df)
    
    if column in ["Backward Transfer", "Forward Transfer"]:
        best_idx = df[column].idxmax()
        worst_idx = df[column].idxmin()
    else:
        best_idx = df[column].idxmax()
        worst_idx = df[column].idxmin()
    
    styles = []
    for idx in df.index:
        if idx == best_idx:
            styles.append("background-color: #90EE90; color: black")
        elif idx == worst_idx:
            styles.append("background-color: #FFB6C1; color: black")
        else:
            styles.append("")
    return styles

st.title("RESEARCH - Continual Learning Benchmark")

st.sidebar.title("Configuration")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

available_runs = ["sample_run"]
if results_dir.exists():
    available_runs.extend([f.stem for f in results_dir.glob("*.json")])

run_selector = st.sidebar.selectbox("Select Run", available_runs)
dataset_selector = st.sidebar.selectbox("Dataset", ["MNIST", "CIFAR-10", "Fashion-MNIST", "SVHN"])

available_methods = ["Naive", "EWC", "GEM", "Replay", "ResearchHybrid"]
selected_methods = st.sidebar.multiselect("Methods to Display", available_methods, default=available_methods)

if run_selector == "sample_run":
    run_data = create_dummy_results(run_selector)
else:
    run_file = results_dir / f"{run_selector}.json"
    if run_file.exists():
        with open(run_file) as f:
            run_data = json.load(f)
    else:
        run_data = create_dummy_results(run_selector)

st.header("Section 1: Leaderboard")

if "methods" in run_data:
    leaderboard_data = []
    for method_name, method_data in run_data["methods"].items():
        if method_name in selected_methods:
            results_matrix = method_data.get("results_matrix")
            if results_matrix:
                import numpy as np
                metrics = compute_metrics(np.array(results_matrix), method_name)
                leaderboard_data.append(metrics)
    
    if leaderboard_data:
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        styled_df = leaderboard_df.style.apply(lambda x: highlight_best_worst(leaderboard_df, x.name), subset=leaderboard_df.columns[1:], axis=0)
        
        st.dataframe(styled_df, use_container_width=True, height=300)
    else:
        st.warning("No results available for the selected methods.")
else:
    st.warning("No method results found in the data.")

st.markdown("---")

st.header("Section 2: Training Curves")

if "methods" in run_data:
    training_data = []
    
    for method_name, method_data in run_data["methods"].items():
        if method_name in selected_methods:
            training_curves = method_data.get("training_curves", {})
            for task_id, curve_data in training_curves.items():
                val_accs = curve_data.get("val_acc", [])
                for t, acc in enumerate(val_accs):
                    training_data.append({
                        "Method": method_name,
                        "Tasks Trained": int(t) + 1,
                        "Avg Accuracy": acc
                    })
    
    if training_data:
        training_df = pd.DataFrame(training_data)
        
        fig = px.line(
            training_df,
            x="Tasks Trained",
            y="Avg Accuracy",
            color="Method",
            markers=True,
            title="Accuracy per Task Over Training Time",
            color_discrete_map={
                "Naive": "#FF6B6B",
                "EWC": "#4ECDC4",
                "GEM": "#45B7D1",
                "Replay": "#96CEB4",
                "ResearchHybrid": "#9B59B6"
            }
        )
        
        fig.update_layout(
            xaxis_title="Tasks Trained So Far",
            yaxis_title="Average Accuracy",
            legend_title="Method",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training curve data available.")
else:
    st.warning("No method results found in the data.")

st.markdown("---")

st.header("Section 3: Forgetting Heatmap")

if "methods" in run_data:
    heatmap_data = []
    
    for method_name, method_data in run_data["methods"].items():
        if method_name in selected_methods:
            results_matrix = method_data.get("results_matrix")
            if results_matrix:
                import numpy as np
                matrix = np.array(results_matrix)
                num_tasks = matrix.shape[0]
                
                for i in range(num_tasks):
                    for j in range(num_tasks):
                        heatmap_data.append({
                            "Method": method_name,
                            "Task Trained Up To": i + 1,
                            "Task Evaluated": j + 1,
                            "Accuracy": matrix[i, j]
                        })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        
        methods_list = list(selected_methods)
        cols = st.columns(len(methods_list))
        
        for idx, method in enumerate(methods_list):
            method_heatmap = heatmap_df[heatmap_df["Method"] == method]
            pivot = method_heatmap.pivot(index="Task Trained Up To", columns="Task Evaluated", values="Accuracy")
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
                hovertemplate="Task Trained: %{y}<br>Task Evaluated: %{x}<br>Accuracy: %{z:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{method}",
                xaxis_title="Task Evaluated",
                yaxis_title="Task Trained Up To",
                width=350,
                height=300
            )
            
            cols[idx].plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No heatmap data available.")
else:
    st.warning("No method results found in the data.")

st.markdown("---")

st.header("Section 4: Radar Chart")

if "methods" in run_data and leaderboard_data:
    metrics = ["Avg Accuracy", "Backward Transfer", "Forward Transfer", "Forgetting"]
    
    fig = go.Figure()
    
    colors = {
        "Naive": "#FF6B6B",
        "EWC": "#4ECDC4",
        "GEM": "#45B7D1",
        "Replay": "#96CEB4",
        "ResearchHybrid": "#9B59B6"
    }
    
    for method_name in selected_methods:
        method_row = next((row for row in leaderboard_data if row["Method"] == method_name), None)
        if method_row:
            values = [method_row[m] for m in metrics]
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=method_name,
                line_color=colors.get(method_name, "#000000")
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-0.5, 1]
            )
        ),
        showlegend=True,
        title="Method Comparison Across Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for radar chart.")

st.markdown("---")

st.header("Section 5: Method Explainer")

explainer_method = st.selectbox("Select a Method", selected_methods)

if explainer_method in METHOD_DESCRIPTIONS:
    st.subheader(f"{explainer_method} Description")
    st.write(METHOD_DESCRIPTIONS[explainer_method])
    
    st.subheader("Hyperparameters")
    if explainer_method in METHOD_HYPERPARAMS:
        hyperparams_df = pd.DataFrame([METHOD_HYPERPARAMS[explainer_method]])
        st.dataframe(hyperparams_df, use_container_width=True, hide_index=True)
    else:
        st.write("No hyperparameters available.")
else:
    st.warning("No description available for this method.")

st.markdown("---")
st.markdown("---")