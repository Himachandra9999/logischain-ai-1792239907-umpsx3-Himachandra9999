
import numpy as np
import pandas as pd


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic financial and supply-chain features from available columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    result = df.copy()

    # Calculate cash conversion cycle when its components are available.
    ccc_columns = {"days_inventory_outstanding", "days_sales_outstanding", "days_payables_outstanding"}
    if ccc_columns.issubset(result.columns):
        result["cash_conversion_cycle"] = (
            result["days_inventory_outstanding"]
            + result["days_sales_outstanding"]
            - result["days_payables_outstanding"]
        )

    # Calculate debt-to-equity ratio when the required values exist.
    if {"total_debt", "total_equity"}.issubset(result.columns):
        equity = pd.to_numeric(result["total_equity"], errors="coerce")
        debt = pd.to_numeric(result["total_debt"], errors="coerce")
        result["debt_to_equity"] = np.where(
            equity != 0,
            debt / equity,
            np.nan,
        )

    # Calculate freight cost per unit when both values exist.
    if {"freight_cost", "shipment_quantity"}.issubset(result.columns):
        quantity = pd.to_numeric(result["shipment_quantity"], errors="coerce")
        freight = pd.to_numeric(result["freight_cost"], errors="coerce")
        result["freight_cost_per_unit"] = np.where(
            quantity > 0,
            freight / quantity,
            np.nan,
        )

    return result
