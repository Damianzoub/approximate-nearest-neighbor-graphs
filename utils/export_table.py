import pandas as pd

def export_for_thesis(df,base_name="ann_results"):
    df.reset_index().to_csv(f"{base_name}.csv",index=False)
    #latex table
    latex = df.reset_index().to_latex(
        index=False,
        float_format="%.4f",
        caption = "ANN Evaluation Results (QPS and Recall@k)",
        label = "tab:ann_results"
    )

    with open(f"{base_name}.text","w",encoding="utf-8") as f:
        f.write(latex)
    
    return f"{base_name}.csv", f"{base_name}.tex"
