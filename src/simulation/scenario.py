
def simulate_logistics_disruption(
    shipment_count: int,
    baseline_delay_days: float,
    baseline_freight_cost: float,
    delay_increase_percent: float,
    freight_increase_percent: float,
) -> dict:
    """Estimate shipment delays and freight costs under a disruption."""
    if shipment_count < 0:
        raise ValueError("Shipment count must not be negative.")

    if baseline_delay_days < 0 or baseline_freight_cost < 0:
        raise ValueError("Baseline delay and freight cost must not be negative.")

    if delay_increase_percent < 0 or freight_increase_percent < 0:
        raise ValueError("Increase percentages must not be negative.")

    projected_delay = baseline_delay_days * (
        1 + delay_increase_percent / 100
    )
    projected_freight = baseline_freight_cost * (
        1 + freight_increase_percent / 100
    )

    return {
        "shipment_count": shipment_count,
        "baseline_delay_days": baseline_delay_days,
        "projected_delay_days": projected_delay,
        "delay_increase_days": projected_delay - baseline_delay_days,
        "baseline_freight_cost": baseline_freight_cost,
        "projected_freight_cost": projected_freight,
        "additional_freight_cost_per_shipment": (
            projected_freight - baseline_freight_cost
        ),
        "total_additional_freight_cost": (
            shipment_count * (projected_freight - baseline_freight_cost)
        ),
    }
