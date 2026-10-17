
def calculate_risk_adjusted_rate(
    base_rate: float,
    risk_premium: float,
) -> float:
    """Calculate an annual financing rate as a percentage."""
    if base_rate < 0 or risk_premium < 0:
        raise ValueError("Rates must not be negative.")

    return base_rate + risk_premium


def calculate_financing_cost(
    principal: float,
    annual_rate_percent: float,
    duration_days: int,
) -> float:
    """Estimate simple financing cost using a 365-day year."""
    if principal < 0:
        raise ValueError("Principal must not be negative.")
    if annual_rate_percent < 0:
        raise ValueError("Annual rate must not be negative.")
    if duration_days < 0:
        raise ValueError("Duration must not be negative.")

    return principal * (annual_rate_percent / 100) * (duration_days / 365)
