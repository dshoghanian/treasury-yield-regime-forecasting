
from typing import Dict
import pandas as pd

def apply_shock(df: pd.DataFrame, shocks: Dict[str, float]) -> pd.DataFrame:
    df2 = df.copy()
    for col, bump in shocks.items():
        if col in df2.columns:
            df2[col] = df2[col] + bump
    return df2
