import re
import json
import pandas as pd
from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

EVALUATION_PROMPT = """
You are a financial data Q&A evaluator.

You are given:
- A **question** generated from a document chunk.
- The **document chunk** (ground truth source).
- A **model-generated answer** to the question.

Your job is to score the model’s answer by carefully comparing it to the document chunk.

Use the following rubric for each category:

---
**Factual Correctness**
- 5 = All facts are fully correct and consistent with the chunk.
- 4 = Minor factual inaccuracies but mostly correct.
- 3 = Some factual inaccuracies, partly correct.
- 2 = Major factual mistakes, mostly incorrect.
- 1 = Completely factually wrong.

---
**Completeness**
- 5 = Fully answers the question with all key details.
- 4 = Mostly complete, missing minor details.
- 3 = Partially complete, missing important parts.
- 2 = Mostly incomplete, only touches on part of the question.
- 1 = Completely incomplete.

---
**Clarity**
- 5 = Clear, precise, and easy to understand.
- 4 = Mostly clear, with minor awkwardness.
- 3 = Understandable but somewhat confusing or vague.
- 2 = Hard to understand or poorly phrased.
- 1 = Completely unclear or nonsensical.

---
**Response Format**
Return ONLY this JSON (no extra explanation):
{
    "factual_correctness_score": [1-5],
    "completeness_score": [1-5],
    "clarity_score": [1-5],
    "comments": "A brief explanation (1-2 sentences) why you assigned these scores."
}
"""

def evaluate_answer(question, chunk, answer):
    """Uses Perplexity LLM to score a model answer"""
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {
                "role": "user",
                "content": f""" 
Please evaluate the following answer based on the provided question and document chunk.
Return ONLY a valid JSON object.
                
Question: {question}
                
Document Chunk: {chunk}
                
Model Answer: {answer}
"""
            }
        ]
    )

    response_content = response.choices[0].message.content.strip()
    try:
        return json.loads(response_content)
    except:
        # fallback fix if double keys break JSON
        response_content_cleaned = re.sub(
            r'(,\s*")(\w+_score)":\s*\d,\s*"\2":\s*\d',
            lambda m: f',{m.group(2)}": {m.group(0).split(":")[-1]}',
            response_content
        )
        return json.loads(response_content_cleaned)

def statistics(df, highlight=True):
    """Compute descriptive statistics for evaluation scores"""
    # Ensure numeric columns
    df = df.copy()
    df["overall_score"] = df[
        ["evaluation_factual_correctness_score", "evaluation_completeness_score", "evaluation_clarity_score"]
    ].mean(axis=1)

    cols = [
        "evaluation_factual_correctness_score",
        "evaluation_completeness_score",
        "evaluation_clarity_score",
        "overall_score",
    ]

    # Describe and calculate median
    stats = (
        df[cols]
        .describe()
        .T.assign(median=lambda x: x["50%"])
        .loc[:, ["count", "mean", "std", "min", "median", "max"]]
        .round(2)
    )
    stats["count"] = stats["count"].astype(int)

    if not highlight:
        return stats

    # Highlight the mean column
    dark_blue = "#003366"
    return (
        stats.style
        .set_properties(
            subset=["mean"],
            **{
                "background-color": dark_blue,
                "color": "white",
                "font-weight": "bold",
            }
        )
        .format("{:.2f}")
    )

def evaluate_dataframe(df):
    """Evaluate all rows in a combined dataframe with columns: question, chunk, answer"""
    rows = []
    for _, row in df.iterrows():
        success = False
        while not success:
            try:
                evaluation = evaluate_answer(row["question"], row["chunk"], row["answer"])
                success = True
            except Exception as e:
                print(f"Retrying due to error: {e}")
                continue

        result = row.to_dict()
        for k, v in evaluation.items():
            result[f"evaluation_{k}"] = v
        rows.append(result)

    return pd.DataFrame(rows)