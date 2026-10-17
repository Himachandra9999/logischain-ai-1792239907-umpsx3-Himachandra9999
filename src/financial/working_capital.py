
def calculate_cash_conversion_cycle(
    days_inventory_outstanding: float,
    days_sales_outstanding: float,
    days_payables_outstanding: float,
) -> float:
    """Calculate the cash conversion cycle in days."""
    return (
        days_inventory_outstanding
        + days_sales_outstanding
        - days_payables_outstanding
    )


def calculate_working_capital(
    current_assets: float,
    current_liabilities: float,
) -> float:
    """Calculate net working capital."""
    return current_assets - current_liabilities


def calculate_current_ratio(
    current_assets: float,
    current_liabilities: float,
) -> float:
    """Calculate the current ratio, or return infinity if liabilities are zero."""
    if current_liabilities == 0:
        raise ValueError("Current liabilities must not be zero.")

    return current_assets / current_liabilities

