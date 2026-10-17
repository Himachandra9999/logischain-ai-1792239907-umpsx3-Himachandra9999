import plotly.express as px
import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_FILE = PROJECT_ROOT / "evaluation" / "model_comparison.json"

st.set_page_config(
    page_title="LogisChain AI",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 LogisChain AI")
st.subheader("Predictive Trade Finance & Logistics Valuation")

st.info(
    "Prototype dashboard: credit-risk results use the UCI German Credit "
    "benchmark. They are not validated for trade-finance lending decisions."
)

st.markdown(
    """
    Explore the project's current machine-learning results and financial
    and logistics analytics foundations.
    """
)


if EVALUATION_FILE.is_file():
    with EVALUATION_FILE.open(encoding="utf-8") as file:
        comparison = pd.DataFrame(json.load(file))

    st.header("📊 Credit-Risk Model Comparison")

    metric_columns = st.columns(3)

    best_accuracy = comparison.loc[comparison["accuracy"].idxmax()]
    best_auc = comparison.loc[comparison["roc_auc"].idxmax()]
    best_recall = comparison.loc[
        comparison["bad_credit_recall"].idxmax()
    ]

    metric_columns[0].metric(
        "Best Accuracy",
        f"{best_accuracy['accuracy']:.1%}",
        best_accuracy["model"],
    )
    metric_columns[1].metric(
        "Best ROC-AUC",
        f"{best_auc['roc_auc']:.3f}",
        best_auc["model"],
    )
    metric_columns[2].metric(
        "Best Bad-Credit Recall",
        f"{best_recall['bad_credit_recall']:.1%}",
        best_recall["model"],
    )

    
    st.subheader("Model Metrics")

    chart_data = comparison.melt(
        id_vars="model",
        value_vars=[
            "accuracy",
            "roc_auc",
            "bad_credit_recall",
            "bad_credit_f1",
        ],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="model",
        barmode="group",
        text="Score",
        title="Credit-Risk Model Performance",
    )

    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig.update_layout(yaxis_title="Score", xaxis_title="Metric")

    st.plotly_chart(fig, use_container_width=True)


    st.caption(
        "Recall measures the proportion of bad-credit cases detected. "
        "Metrics are based on a held-out test set of 200 records."
    )

else:
    st.warning(
        "Model comparison results are missing. Run "
        "`python -m src.models.compare_models` in the terminal first."
    )


st.divider()
st.header("🏗️ Project Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Credit-Risk Analytics**")
    st.write("Benchmark classification and model evaluation.")

with col2:
    st.markdown("**Working-Capital Features**")
    st.write("Cash conversion cycle and debt-to-equity calculations.")

with col3:
    st.markdown("**Logistics Simulation**")
    st.write("Estimate the impact of shipment disruptions on delays and costs.")

st.divider()
st.caption("LogisChain AI | ZeTheta Project 2 | Prototype")

from src.simulation.scenario import simulate_logistics_disruption

st.divider()
st.header("🚚 Logistics Disruption Simulator")

st.write(
    "Estimate how a logistics disruption could affect delivery delays "
    "and freight costs. These are scenario estimates, not live forecasts."
)

with st.form("logistics_simulation_form"):
    shipment_count = st.number_input(
        "Number of shipments",
        min_value=0,
        value=100,
        step=10,
    )

    baseline_delay_days = st.number_input(
        "Baseline delivery delay (days)",
        min_value=0.0,
        value=3.0,
        step=0.5,
    )

    baseline_freight_cost = st.number_input(
        "Baseline freight cost per shipment (₹)",
        min_value=0.0,
        value=5000.0,
        step=500.0,
    )

    delay_increase_percent = st.slider(
        "Disruption-related delay increase (%)",
        min_value=0,
        max_value=200,
        value=25,
    )

    freight_increase_percent = st.slider(
        "Disruption-related freight increase (%)",
        min_value=0,
        max_value=200,
        value=20,
    )

    run_simulation = st.form_submit_button("Run Simulation")

if run_simulation:
    try:
        simulation = simulate_logistics_disruption(
            shipment_count=int(shipment_count),
            baseline_delay_days=baseline_delay_days,
            baseline_freight_cost=baseline_freight_cost,
            delay_increase_percent=delay_increase_percent,
            freight_increase_percent=freight_increase_percent,
        )

        st.subheader("Simulation Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Projected delay",
            f"{simulation['projected_delay_days']:.2f} days",
            delta=(
                f"+{simulation['delay_increase_days']:.2f} days"
                if simulation["delay_increase_days"] > 0
                else "No increase"
            ),
            delta_color="inverse",
        )

        result_col2.metric(
            "Projected freight / shipment",
            f"₹{simulation['projected_freight_cost']:,.2f}",
            delta=(
                f"+₹{simulation['additional_freight_cost_per_shipment']:,.2f}"
                if simulation["additional_freight_cost_per_shipment"] > 0
                else "No increase"
            ),
            delta_color="inverse",
        )

        result_col3.metric(
            "Total additional freight",
            f"₹{simulation['total_additional_freight_cost']:,.2f}",
        )

        comparison = pd.DataFrame(
            {
                "Scenario": ["Baseline", "After disruption"],
                "Delay (days)": [
                    simulation["baseline_delay_days"],
                    simulation["projected_delay_days"],
                ],
                "Freight per shipment (₹)": [
                    simulation["baseline_freight_cost"],
                    simulation["projected_freight_cost"],
                ],
            }
        )

        st.dataframe(comparison, use_container_width=True, hide_index=True)

    except ValueError as error:
        st.error(str(error))

from src.financial.working_capital import (
    calculate_cash_conversion_cycle,
    calculate_working_capital,
    calculate_current_ratio,
)

st.divider()
st.header("💰 Working-Capital Calculator")

st.write(
    "Enter illustrative financial and operating figures to calculate "
    "working-capital indicators. These calculations are not financial advice."
)

with st.form("working_capital_form"):
    st.subheader("Operating Cycle")

    wc_col1, wc_col2, wc_col3 = st.columns(3)

    with wc_col1:
        inventory_days = st.number_input(
            "Days Inventory Outstanding",
            min_value=0.0,
            value=45.0,
            step=1.0,
        )

    with wc_col2:
        sales_days = st.number_input(
            "Days Sales Outstanding",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )

    with wc_col3:
        payable_days = st.number_input(
            "Days Payables Outstanding",
            min_value=0.0,
            value=25.0,
            step=1.0,
        )

    st.subheader("Balance-Sheet Figures")

    finance_col1, finance_col2 = st.columns(2)

    with finance_col1:
        current_assets = st.number_input(
            "Current Assets (₹)",
            min_value=0.0,
            value=500000.0,
            step=10000.0,
        )

    with finance_col2:
        current_liabilities = st.number_input(
            "Current Liabilities (₹)",
            min_value=0.0,
            value=300000.0,
            step=10000.0,
        )

    calculate_wc = st.form_submit_button("Calculate Working Capital")

if calculate_wc:
    try:
        ccc = calculate_cash_conversion_cycle(
            inventory_days,
            sales_days,
            payable_days,
        )

        net_working_capital = calculate_working_capital(
            current_assets,
            current_liabilities,
        )

        current_ratio = calculate_current_ratio(
            current_assets,
            current_liabilities,
        )

        st.subheader("Working-Capital Results")

        wc_result1, wc_result2, wc_result3 = st.columns(3)

        wc_result1.metric("Cash Conversion Cycle", f"{ccc:.1f} days")
        wc_result2.metric(
            "Net Working Capital",
            f"₹{net_working_capital:,.2f}",
        )
        wc_result3.metric("Current Ratio", f"{current_ratio:.2f}×")

        if ccc < 0:
            st.info(
                "The cash conversion cycle is negative. This can occur "
                "when a business collects cash before paying suppliers."
            )
        if net_working_capital < 0:
            st.warning(
                "Net working capital is negative. Review short-term "
                "assets and liabilities in the context of the business."
            )

    except ValueError as error:
        st.error(str(error))

import joblib

from src.data.german_credit import GERMAN_CREDIT_COLUMNS


st.divider()
st.header("🧾 Credit-Risk Prediction")
st.warning(
    "Research demonstration only. This prediction uses the UCI German Credit "
    "benchmark and must not be used to approve or reject real applicants."
)

MODEL_FILE = PROJECT_ROOT / "models" / "german_credit_rf.joblib"

if MODEL_FILE.is_file():
    credit_model = joblib.load(MODEL_FILE)

    # Example categories from the UCI German Credit dataset.
    categories = {
        "checking_account_status": ["A11", "A12", "A13", "A14"],
        "credit_history": ["A30", "A31", "A32", "A33", "A34"],
        "purpose": [
            "A40", "A41", "A42", "A43", "A44",
            "A45", "A46", "A47", "A48", "A49", "A410"
        ],
        "savings_status": ["A61", "A62", "A63", "A64", "A65"],
        "employment_since": ["A71", "A72", "A73", "A74", "A75"],
        "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
        "other_debtors": ["A101", "A102", "A103"],
        "property": ["A121", "A122", "A123", "A124"],
        "other_installment_plans": ["A141", "A142", "A143"],
        "housing": ["A151", "A152", "A153"],
        "job": ["A171", "A172", "A173", "A174"],
        "telephone": ["A191", "A192"],
        "foreign_worker": ["A201", "A202"],
    }

    numeric_defaults = {
        "duration_months": (12, 1, 72),
        "credit_amount": (2000, 1, 200000),
        "installment_rate": (2, 1, 4),
        "residence_since": (2, 1, 4),
        "age_years": (35, 18, 100),
        "existing_credits": (1, 1, 4),
        "dependents": (1, 1, 2),
    }

    with st.form("credit_prediction_form"):
        st.write("Enter sample applicant information.")

        applicant = {}

        category_columns = st.columns(2)

        for index, (column, options) in enumerate(categories.items()):
            with category_columns[index % 2]:
                applicant[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    options,
                    key=f"credit_{column}",
                )

        st.write("Numeric details")
        numeric_columns = st.columns(2)

        for index, (column, values) in enumerate(numeric_defaults.items()):
            default, minimum, maximum = values
            with numeric_columns[index % 2]:
                applicant[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1,
                    key=f"credit_{column}",
                )

        predict_credit = st.form_submit_button("Predict Benchmark Credit Risk")

    if predict_credit:
        # Preserve the exact column order used in model training.
        input_df = pd.DataFrame(
            [{column: applicant[column] for column in GERMAN_CREDIT_COLUMNS[:-1]}]
        )

        prediction = int(credit_model.predict(input_df)[0])
        probabilities = credit_model.predict_proba(input_df)[0]
        bad_credit_probability = float(probabilities[1])

        if prediction == 1:
            st.error("Benchmark prediction: Bad Credit")
        else:
            st.success("Benchmark prediction: Good Credit")

        st.metric(
            "Estimated bad-credit probability",
            f"{bad_credit_probability:.1%}",
        )

        st.caption(
            "This is a model estimate on a historical benchmark, not a "
            "verified probability of default for a real applicant."
        )
else:
    st.error(
        "The trained model file was not found. Run "
        "`python -m src.models.train_credit_model` first."
    )
