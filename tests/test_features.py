
import pandas as pd

from src.features.engineering import add_financial_features


def test_cash_conversion_cycle_feature():
    df = pd.DataFrame(
        {
            "days_inventory_outstanding": [30],
            "days_sales_outstanding": [40],
            "days_payables_outstanding": [20],
        }
    )

    result = add_financial_features(df)

    assert result.loc[0, "cash_conversion_cycle"] == 50
    # The input DataFrame should remain unchanged.
    assert "cash_conversion_cycle" not in df.columns


def test_debt_to_equity_feature():
    df = pd.DataFrame(
        {
            "total_debt": [100, 50],
            "total_equity": [50, 0],
        }
    )

    result = add_financial_features(df)

    assert result.loc[0, "debt_to_equity"] == 2
    assert pd.isna(result.loc[1, "debt_to_equity"])


def test_freight_cost_per_unit_feature():
    df = pd.DataFrame(
        {
            "freight_cost": [600, 200],
            "shipment_quantity": [30, 0],
        }
    )

    result = add_financial_features(df)

    assert result.loc[0, "freight_cost_per_unit"] == 20
    assert pd.isna(result.loc[1, "freight_cost_per_unit"])
