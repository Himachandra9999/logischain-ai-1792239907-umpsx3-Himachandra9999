
import pytest

from src.simulation import simulate_logistics_disruption


def test_disruption_increases_delay_and_freight_cost():
    result = simulate_logistics_disruption(
        shipment_count=100,
        baseline_delay_days=4,
        baseline_freight_cost=2000,
        delay_increase_percent=20,
        freight_increase_percent=15,
    )

    assert result["projected_delay_days"] == pytest.approx(4.8)
    assert result["projected_freight_cost"] == pytest.approx(2300)
    assert result["total_additional_freight_cost"] == pytest.approx(30000)


def test_zero_shipments_has_zero_additional_total_cost():
    result = simulate_logistics_disruption(
        shipment_count=0,
        baseline_delay_days=4,
        baseline_freight_cost=2000,
        delay_increase_percent=20,
        freight_increase_percent=15,
    )

    assert result["total_additional_freight_cost"] == 0


def test_negative_shipment_count_is_rejected():
    with pytest.raises(ValueError):
        simulate_logistics_disruption(
            shipment_count=-1,
            baseline_delay_days=4,
            baseline_freight_cost=2000,
            delay_increase_percent=20,
            freight_increase_percent=15,
        )
