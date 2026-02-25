"""
Sensitivity tests for budget impact estimates.

Tests four assumptions in the baseline model:
1. Income deflation: use RPI-deflated income for historical years
   (instead of current income as proxy for past income)
2. Graduation age: 22 instead of 21
3. Debt variance: log-normal spread around cohort average
4. Early repayment: balances that reach zero stay cleared
   (already handled by np.maximum(0, ...) but income deflation
   changes who clears early)

Run: python sensitivity_test.py
"""

import json
import sys

import numpy as np
from policyengine_uk import Microsimulation
from policyengine_uk.model_api import (
    YEAR,
    GBP,
    Person,
    Variable,
    max_,
    min_,
    select,
    where,
)
from policyengine_uk.system import system
from policyengine_uk.utils.scenario import Scenario
from policyengine_uk.variables.gov.hmrc.student_loans.student_loan_plan import (
    StudentLoanPlan,
)

# Import shared pieces from main script
from generate_budget_impact import (
    YEARS,
    BASE_YEAR,
    PLAN_2_DEBT_LATEST,
    PLAN_2_COHORT_DEBT,
    PLAN_2_HISTORIC_RATES,
    connect_student_loan_variables,
    _get_threshold,
)


def add_balance_tracking_sensitivity(
    sim,
    graduation_age=21,
    deflate_income=False,
    debt_variance_std=0.0,
    random_seed=42,
):
    """Balance tracking with configurable sensitivity parameters.

    Parameters
    ----------
    sim : Microsimulation
    graduation_age : int
        Assumed age at graduation (default 21, test with 22).
    deflate_income : bool
        If True, deflate current income by cumulative RPI to estimate
        historical income for past repayment calculations.
    debt_variance_std : float
        Standard deviation of log-normal multiplier on cohort debt.
        0.0 = no variance (everyone gets the average).
        0.3 = moderate spread (~£30k-£65k range for most borrowers).
    random_seed : int
        Random seed for debt variance reproducibility.
    """

    # Pre-compute cumulative RPI deflators for income backdating
    march_rpi = {
        2015: 0.009, 2016: 0.016, 2017: 0.031, 2018: 0.033,
        2019: 0.024, 2020: 0.026, 2021: 0.015, 2022: 0.090,
        2023: 0.135, 2024: 0.043, 2025: 0.032,
    }
    # cumulative_deflator[years_ago] = product of (1+rpi) from that year to now
    cumulative_deflator = {}
    for years_ago in range(1, 12):
        year = BASE_YEAR - years_ago
        deflator = 1.0
        for y in range(year, BASE_YEAR):
            deflator *= 1 + march_rpi.get(y + 1, 0.03)
        cumulative_deflator[years_ago] = deflator

    # Closure variables for the inner class
    _graduation_age = graduation_age
    _deflate_income = deflate_income
    _debt_variance_std = debt_variance_std
    _random_seed = random_seed
    _cumulative_deflator = cumulative_deflator

    class student_loan_initial_balance(Variable):
        value_type = float
        entity = Person
        label = "Student loan initial balance (sensitivity)"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            plan = person("student_loan_plan", period)
            is_plan_2 = plan == StudentLoanPlan.PLAN_2
            age = person("age", period)
            income = person("adjusted_net_income", period)
            p = parameters(period).gov.hmrc.student_loans

            years_int = np.clip(
                np.round(max_(0, age - _graduation_age)).astype(int),
                0,
                11,
            )

            # Cohort-specific original debt
            grad_year = np.clip(BASE_YEAR - years_int, 2015, 2025)
            cohort_years = sorted(PLAN_2_COHORT_DEBT.keys())
            cohort_debts = [PLAN_2_COHORT_DEBT[y] for y in cohort_years]
            original_debt = np.interp(grad_year, cohort_years, cohort_debts)

            # Add debt variance if configured
            if _debt_variance_std > 0:
                rng = np.random.default_rng(_random_seed)
                multiplier = rng.lognormal(
                    mean=0.0,
                    sigma=_debt_variance_std,
                    size=len(original_debt),
                )
                # Normalise so the mean stays at the cohort average
                multiplier /= np.exp(_debt_variance_std**2 / 2)
                original_debt = original_debt * multiplier

            balance = np.where(
                is_plan_2, original_debt, 0.0
            ).astype(float)

            # Walk forward from graduation to present
            for years_ago in range(11, 0, -1):
                year = BASE_YEAR - years_ago
                graduated = years_int >= years_ago

                if year in PLAN_2_HISTORIC_RATES:
                    rpi, cap = PLAN_2_HISTORIC_RATES[year]

                    # Deflate income to estimate what it was in that year
                    if _deflate_income:
                        hist_income = income / _cumulative_deflator.get(
                            years_ago, 1.0
                        )
                    else:
                        hist_income = income

                    # Interest taper based on (historical) income
                    upper = p.interest_rates.plan_2.upper_threshold
                    taper = np.clip(
                        (hist_income - p.thresholds.plan_2)
                        / (upper - p.thresholds.plan_2),
                        0,
                        1,
                    )

                    # Historical repayment based on (historical) income
                    annual_repay = p.repayment_rate * max_(
                        0, hist_income - p.thresholds.plan_2
                    )

                    if cap is not None and cap < rpi:
                        rate = np.full_like(income, cap)
                    elif cap is not None:
                        rate = np.minimum(rpi + 0.03 * taper, cap)
                    else:
                        rate = rpi + 0.03 * taper

                    new_balance = np.maximum(
                        0, balance * (1 + rate) - annual_repay
                    )
                    balance = np.where(graduated, new_balance, balance)

            return np.where(is_plan_2, balance, 0.0)

    class student_loan_repayment(Variable):
        value_type = float
        entity = Person
        label = "Student loan repayment (sensitivity, balance-aware)"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            plan = person("student_loan_plan", period)
            income = person("adjusted_net_income", period)
            p = parameters(period).gov.hmrc.student_loans

            threshold = _get_threshold(plan, p)
            rate = person("student_loan_repayment_rate", period)
            uncapped = rate * max_(0, income - threshold)

            if period.start.year <= BASE_YEAR:
                prev_balance = person(
                    "student_loan_initial_balance", period
                )
            else:
                prev_balance = person(
                    "student_loan_balance", period.last_year
                )

            interest_rate = person("student_loan_interest_rate", period)
            balance_with_interest = prev_balance * (1 + interest_rate)
            return min_(uncapped, max_(0, balance_with_interest))

    class student_loan_balance(Variable):
        value_type = float
        entity = Person
        label = "Student loan balance (sensitivity)"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            if period.start.year <= BASE_YEAR:
                prev_balance = person(
                    "student_loan_initial_balance", period
                )
            else:
                prev_balance = person(
                    "student_loan_balance", period.last_year
                )

            interest_rate = person("student_loan_interest_rate", period)
            balance_with_interest = prev_balance * (1 + interest_rate)
            repayment = person("student_loan_repayment", period)
            return max_(0, balance_with_interest - repayment)

    class student_loan_interest_accrual(Variable):
        value_type = float
        entity = Person
        label = "Annual interest accrued (sensitivity)"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            if period.start.year <= BASE_YEAR:
                prev_balance = person(
                    "student_loan_initial_balance", period
                )
            else:
                prev_balance = person(
                    "student_loan_balance", period.last_year
                )

            interest_rate = person("student_loan_interest_rate", period)
            return prev_balance * interest_rate

    sim.tax_benefit_system.add_variable(student_loan_initial_balance)
    sim.tax_benefit_system.add_variable(student_loan_balance)
    sim.tax_benefit_system.add_variable(student_loan_interest_accrual)
    sim.tax_benefit_system.update_variable(student_loan_repayment)
    return sim


def run_interest_cap_scenario(label, graduation_age=21, deflate_income=False,
                               debt_variance_std=0.0):
    """Run the interest cap reform under given assumptions and return results."""

    interest_cap_params = {
        "gov.hmrc.student_loans.interest_rates.plan_2.additional_rate": {
            str(y): 0.0 for y in YEARS
        },
    }

    def make_modifier(**kwargs):
        def modifier(sim):
            sim = connect_student_loan_variables(sim)
            sim = add_balance_tracking_sensitivity(sim, **kwargs)
            return sim
        return modifier

    sens_kwargs = dict(
        graduation_age=graduation_age,
        deflate_income=deflate_income,
        debt_variance_std=debt_variance_std,
    )

    print(f"\n--- {label} ---")
    print(f"  graduation_age={graduation_age}, deflate_income={deflate_income}, "
          f"debt_variance_std={debt_variance_std}")

    baseline = Microsimulation(
        scenario=Scenario(
            simulation_modifier=make_modifier(**sens_kwargs),
        )
    )

    reformed = Microsimulation(
        scenario=Scenario(
            parameter_changes=interest_cap_params,
            simulation_modifier=make_modifier(**sens_kwargs),
        )
    )

    # Diagnostics
    init_bal = baseline.calculate("student_loan_initial_balance", period=BASE_YEAR)
    plan = baseline.calculate("student_loan_plan", period=BASE_YEAR)
    weight = baseline.calculate("person_weight", period=BASE_YEAR)
    p2_mask = plan.values == "PLAN_2"
    p2_book = (init_bal.values[p2_mask] * weight.values[p2_mask]).sum() / 1e9
    p2_avg = np.average(init_bal.values[p2_mask], weights=weight.values[p2_mask])
    p2_count = weight.values[p2_mask].sum() / 1e6
    zero_bal = (
        weight.values[p2_mask & (init_bal.values == 0)].sum()
        / weight.values[p2_mask].sum()
        * 100
    )
    print(f"  Loan book: £{p2_book:.0f}bn | {p2_count:.1f}m borrowers | "
          f"avg £{p2_avg:,.0f} | {zero_bal:.1f}% already cleared")

    results = []
    for year in YEARS:
        b_interest = baseline.calculate("student_loan_interest_accrual", period=year)
        r_interest = reformed.calculate("student_loan_interest_accrual", period=year)
        impact = (r_interest - b_interest).sum() / 1e9

        b_repay = baseline.calculate("student_loan_repayment", period=year)
        r_repay = reformed.calculate("student_loan_repayment", period=year)
        repay_impact = (r_repay - b_repay).sum() / 1e9

        results.append({
            "year": year,
            "interest_impact_bn": round(float(impact), 3),
            "repayment_impact_bn": round(float(repay_impact), 3),
        })
        print(f"  {year}: interest £{impact:+.3f}bn | repayment £{repay_impact:+.3f}bn")

    return {
        "label": label,
        "params": sens_kwargs,
        "loan_book_bn": round(p2_book, 1),
        "avg_balance": round(p2_avg),
        "pct_cleared": round(zero_bal, 1),
        "annual_impacts": results,
    }


def main():
    print("=" * 70)
    print("SENSITIVITY TESTS: Interest cap reform (Reform 2)")
    print("=" * 70)

    scenarios = [
        {
            "label": "Original (current income, age 21, no variance)",
            "graduation_age": 21,
            "deflate_income": False,
            "debt_variance_std": 0.0,
        },
        {
            "label": "Test 1: RPI-deflated historical income",
            "graduation_age": 21,
            "deflate_income": True,
            "debt_variance_std": 0.0,
        },
        {
            "label": "Test 2: Graduation age 22",
            "graduation_age": 22,
            "deflate_income": False,
            "debt_variance_std": 0.0,
        },
        {
            "label": "Test 3: Debt variance (std=0.3)",
            "graduation_age": 21,
            "deflate_income": False,
            "debt_variance_std": 0.3,
        },
        {
            "label": "Test 4: All combined (deflated income + age 22 + variance)",
            "graduation_age": 22,
            "deflate_income": True,
            "debt_variance_std": 0.3,
        },
    ]

    all_results = []
    for s in scenarios:
        result = run_interest_cap_scenario(
            label=s["label"],
            graduation_age=s["graduation_age"],
            deflate_income=s["deflate_income"],
            debt_variance_std=s["debt_variance_std"],
        )
        all_results.append(result)

    # Summary comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<55} {'Book':>7} {'Avg bal':>8} {'Cleared':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['label']:<55} £{r['loan_book_bn']}bn  £{r['avg_balance']:,}  {r['pct_cleared']}%")

    print(f"\n{'Scenario':<55}", end="")
    for year in YEARS:
        print(f" {year:>8}", end="")
    print()
    print("-" * (55 + 9 * len(YEARS)))
    for r in all_results:
        print(f"{r['label']:<55}", end="")
        for imp in r["annual_impacts"]:
            print(f" {abs(imp['interest_impact_bn']):>7.2f}", end="")
        print()

    # Save results (convert numpy types to Python native for JSON)
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            native = to_native(obj)
            if native is not obj:
                return native
            return super().default(obj)

    with open("sensitivity_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print("\nResults saved to sensitivity_results.json")


if __name__ == "__main__":
    main()
