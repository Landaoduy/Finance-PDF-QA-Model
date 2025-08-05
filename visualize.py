import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

def plot_score_distributions(df):
    """Plot boxplots for each evaluation metric and overall score"""
    numeric_cols = [
        "evaluation_factual_correctness_score",
        "evaluation_completeness_score",
        "evaluation_clarity_score"
    ]

    df_clean = df.copy()
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean["overall_score"] = df_clean[numeric_cols].mean(axis=1)

    # Melt to long format
    melted_df = df_clean.melt(
        id_vars=["question"],
        value_vars=numeric_cols + ["overall_score"],
        var_name="Metric",
        value_name="Score"
    )

    # Clean label formatting
    melted_df["Metric"] = (
        melted_df["Metric"]
        .str.replace("evaluation_", "", regex=False)
        .str.replace("_score", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    melted_df = melted_df.dropna(subset=["Score"])

    fig = px.box(
        melted_df,
        x="Metric",
        y="Score",
        points="all",
        hover_data=["question"],
        title="Score Distributions per Metric (with Overall Score)",
        height=500
    )

    fig.update_layout(
        yaxis=dict(range=[0, 6], dtick=1),
        xaxis_title="Metric",
        yaxis_title="Score (1–5)"
    )

    fig.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap between evaluation metrics"""
    df = df.copy()

    # Define evaluation columns
    score_cols = [
        "evaluation_factual_correctness_score",
        "evaluation_completeness_score",
        "evaluation_clarity_score"
    ]

    # Ensure numeric format
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add overall score if not already present
    if "overall_score" not in df.columns:
        df["overall_score"] = df[score_cols].mean(axis=1)

    # Compute correlation matrix
    df_corr = df[score_cols + ["overall_score"]].corr().round(2)

    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=df_corr.values,
        x=df_corr.columns.tolist(),
        y=df_corr.index.tolist(),
        annotation_text=df_corr.values,
        colorscale="Blues",
        showscale=True
    )

    fig.update_layout(
        title="Correlation Heatmap of Evaluation Metrics",
        width=600,
        height=600
    )

    fig.show()

def plot_overall_histogram(df, bin_size=0.25):
    """
    Plot a histogram of overall scores (0‒5).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain an 'overall_score' column.
    bin_size : float, optional
        Width of each bin; defaults to 0.25.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """
    df = df.copy()
    if "overall_score" not in df.columns:
        score_cols = [
            "evaluation_factual_correctness_score",
            "evaluation_completeness_score",
            "evaluation_clarity_score"
        ]
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["overall_score"] = df[score_cols].mean(axis=1)

    fig = px.histogram(
        df,
        x="overall_score",
        nbins=int(5 / bin_size),
        title="Distribution of Overall Scores",
        labels={"overall_score": "Overall Score"},
    )

    # Add mean reference line
    mean_val = df["overall_score"].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        bargap=0.05,
        xaxis=dict(title="Overall Score", dtick=0.5),
        yaxis_title="Count"
    )

    fig.show()
    return fig