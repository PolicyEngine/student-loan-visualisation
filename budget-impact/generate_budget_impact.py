"""
Generate budgetary impact of student loan reforms using PolicyEngine UK microsimulation.

Reform 1 — Cancel the threshold freeze:
  Measured via PE microsimulation. Changing the repayment threshold changes
  annual repayment cash flows (repayment = 9% × max(0, income − threshold)).
  Baseline: current law (frozen at £29,385 for 2027-2029).
  Reform: uprate threshold with RPI each year.

Reform 2 — Cap interest at RPI:
  Measured via PE microsimulation with balance tracking. The fiscal cost
  is the lost interest income on the student loan book: interest accrual
  increases the value of the government's loan asset, and capping at RPI
  wipes out the 0-3% margin. Initial balances estimated per-person using
  PE microdata (age, income, plan type) and actual historical March RPI
  values from SLC/ONS. Average original debt £45,600 from DfE forecasts.

Requires: policyengine-uk >= 2.65.0
"""

import json

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

YEARS = [2026, 2027, 2028, 2029, 2030]
BASE_YEAR = YEARS[0]

# Average Plan 2 debt at graduation: £45,600 for the 2022-23 final cohort.
# Earlier cohorts borrowed roughly the same in real terms, so we deflate
# backward using March RPI to get each cohort's nominal original debt.
#
# Source: DfE Student Loan Forecasts for England 2022-23
# https://explore-education-statistics.service.gov.uk/find-statistics/student-loan-forecasts-for-england/2022-23
PLAN_2_DEBT_LATEST = 45_600  # Last Plan 2 cohort (graduated ~2025)


def _build_cohort_debt_schedule():
    """Deflate £45,600 backward by March RPI to get each cohort's nominal debt."""
    # March RPI values used for deflation (same source as PLAN_2_HISTORIC_RATES)
    march_rpi = {
        2015: 0.009, 2016: 0.016, 2017: 0.031, 2018: 0.033,
        2019: 0.024, 2020: 0.026, 2021: 0.015, 2022: 0.090,
        2023: 0.135, 2024: 0.043, 2025: 0.032,
    }
    # Work backward from 2025: debt_{y-1} = debt_y / (1 + rpi_y)
    debt = {2025: float(PLAN_2_DEBT_LATEST)}
    for y in range(2024, 2014, -1):
        rpi = march_rpi.get(y + 1, 0.03)
        debt[y] = debt[y + 1] / (1 + rpi)
    return debt


PLAN_2_COHORT_DEBT = _build_cohort_debt_schedule()

GRADUATION_AGE = 21

# Historical March RPI values and government rate caps for Plan 2 interest.
# Format: (march_rpi, max_rate_cap_or_None)
# Normal years: rate = RPI + 0.03 × taper (taper 0–1 based on income)
# Capped years: rate = min(cap, RPI + 0.03 × taper).  In 2022-23 the cap
# was BELOW RPI itself, so everyone got the flat cap regardless of income.
#
# Source: SLC, Income Contingent Repayment Plans, Interest Rates & Calculations
# https://www.gov.uk/government/statistics/student-loans-in-england-2024-to-2025/income-contingent-student-loan-repayment-plans-interest-rates-and-calculations-england
# ONS RPI long-run series (CDSI):
# https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/cdsi/mm23
PLAN_2_HISTORIC_RATES = {
    #       (March RPI,  cap or None)
    2015: (0.009, None),   # Sep 15–Aug 16: 0.9%–3.9%
    2016: (0.016, None),   # Sep 16–Aug 17: 1.6%–4.6%
    2017: (0.031, None),   # Sep 17–Aug 18: 3.1%–6.1%
    2018: (0.033, None),   # Sep 18–Aug 19: 3.3%–6.3%
    2019: (0.024, None),   # Sep 19–Aug 20: 2.4%–5.4%
    2020: (0.026, None),   # Sep 20–Aug 21: 2.6%–5.6%
    2021: (0.015, None),   # Sep 21–Aug 22: 1.5%–4.5%
    2022: (0.090, 0.069),  # Sep 22–Aug 23: RPI 9%, capped at 6.9%
    2023: (0.135, 0.077),  # Sep 23–Aug 24: RPI 13.5%, capped at 7.7%
    2024: (0.043, None),   # Sep 24–Aug 25: 4.3%–7.3%
    2025: (0.032, None),   # Sep 25–Aug 26: 3.2%–6.2%
}


# ─── Shared modifier: connect modelled repayments to revenue totals ──────────


def connect_student_loan_variables(sim):
    """Swap FRS-reported student_loan_repayments for PE-modelled variable."""
    sim.tax_benefit_system.variables["gov_tax"].adds.remove(
        "student_loan_repayments"
    )
    sim.tax_benefit_system.variables["gov_tax"].adds.append(
        "student_loan_repayment"
    )
    sim.tax_benefit_system.variables["household_tax"].adds.remove(
        "student_loan_repayments"
    )
    sim.tax_benefit_system.variables["household_tax"].adds.append(
        "student_loan_repayment"
    )
    sim.tax_benefit_system.variables[
        "hbai_household_net_income"
    ].subtracts.append("student_loan_repayment")
    sim.tax_benefit_system.variables["hbai_household_net_income"].adds.append(
        "student_loan_repayments"
    )
    return sim


# ─── Balance-tracking modifier ───────────────────────────────────────────────


def _get_threshold(plan, p):
    """Get repayment threshold for a given plan type (vectorised)."""
    return select(
        [
            plan == StudentLoanPlan.PLAN_1,
            plan == StudentLoanPlan.PLAN_2,
            plan == StudentLoanPlan.PLAN_4,
            plan == StudentLoanPlan.PLAN_5,
            plan == StudentLoanPlan.POSTGRADUATE,
        ],
        [
            p.thresholds.plan_1,
            p.thresholds.plan_2,
            p.thresholds.plan_4,
            p.thresholds.plan_5,
            p.thresholds.postgraduate,
        ],
        default=np.inf,
    )


def add_balance_tracking(sim):
    """Add loan balance tracking so repayments stop once the loan is cleared.

    Adds two new variables and overrides student_loan_repayment:
    - student_loan_initial_balance: assumed starting balance (£45k avg)
    - student_loan_balance: year-on-year balance after interest and repayment
    - student_loan_repayment: capped at remaining balance
    """

    class student_loan_initial_balance(Variable):
        value_type = float
        entity = Person
        label = "Student loan initial balance at start of simulation"
        documentation = (
            "Estimated remaining Plan 2 balance using PE microdata "
            "(age, income, plan type) and actual historical March RPI "
            "values from SLC/ONS (including 2022-23 caps). Simulates "
            "balance year-by-year: interest accrues, then repayment "
            "is subtracted, for each year since graduation."
        )
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            plan = person("student_loan_plan", period)
            is_plan_2 = plan == StudentLoanPlan.PLAN_2
            age = person("age", period)
            income = person("adjusted_net_income", period)
            p = parameters(period).gov.hmrc.student_loans

            # Years since graduation (Plan 2: earliest grads 2015)
            years_int = np.clip(
                np.round(max_(0, age - GRADUATION_AGE)).astype(int),
                0,
                11,
            )

            # Per-person interest taper (from PE income data)
            upper = p.interest_rates.plan_2.upper_threshold
            taper = np.clip(
                (income - p.thresholds.plan_2)
                / (upper - p.thresholds.plan_2),
                0,
                1,
            )

            # Annual repayment (using current income as proxy)
            annual_repay = p.repayment_rate * max_(
                0, income - p.thresholds.plan_2
            )

            # Cohort-specific original debt: deflated by RPI from £45,600.
            grad_year = np.clip(BASE_YEAR - years_int, 2015, 2025)
            cohort_years = sorted(PLAN_2_COHORT_DEBT.keys())
            cohort_debts = [PLAN_2_COHORT_DEBT[y] for y in cohort_years]
            original_debt = np.interp(grad_year, cohort_years, cohort_debts)

            # Simulate balance year-by-year from graduation to 2025.
            # Walk forward: 11 years ago → 1 year ago.
            balance = np.where(
                is_plan_2, original_debt, 0.0
            ).astype(float)

            for years_ago in range(11, 0, -1):
                year = BASE_YEAR - years_ago
                graduated = years_int >= years_ago

                if year in PLAN_2_HISTORIC_RATES:
                    rpi, cap = PLAN_2_HISTORIC_RATES[year]
                    if cap is not None and cap < rpi:
                        # Cap below RPI: flat rate for everyone
                        rate = np.full_like(income, cap)
                    elif cap is not None:
                        # Cap above RPI: taper applies, capped
                        rate = np.minimum(
                            rpi + 0.03 * taper, cap
                        )
                    else:
                        # Normal year: RPI + margin × taper
                        rate = rpi + 0.03 * taper

                    new_balance = np.maximum(
                        0, balance * (1 + rate) - annual_repay
                    )
                    balance = np.where(
                        graduated, new_balance, balance
                    )

            return np.where(is_plan_2, balance, 0.0)

    class student_loan_repayment(Variable):
        value_type = float
        entity = Person
        label = "Student loan repayment (modelled, balance-aware)"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):
            plan = person("student_loan_plan", period)
            income = person("adjusted_net_income", period)
            p = parameters(period).gov.hmrc.student_loans

            threshold = _get_threshold(plan, p)
            rate = person("student_loan_repayment_rate", period)
            uncapped = rate * max_(0, income - threshold)

            # Get balance before this year's repayment
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
        label = "Student loan balance at end of year"
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
        label = "Annual interest accrued on student loan balance"
        documentation = (
            "Interest income earned by the government on the student "
            "loan book. Equal to interest_rate × start-of-year balance."
        )
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


def connect_and_track_balances(sim):
    """Combined modifier: connect variables + add balance tracking."""
    sim = connect_student_loan_variables(sim)
    sim = add_balance_tracking(sim)
    return sim


# ─── Reform 1: Cancel the threshold freeze ───────────────────────────────────


def calculate_uprated_thresholds():
    """Calculate what Plan 2 thresholds would be with RPI uprating (no freeze)."""
    params = system.parameters
    rpi_index = params.gov.economic_assumptions.indices.obr.rpi

    base_2026 = 29_385
    rpi_2026 = rpi_index("2026-04-06")

    return {
        "gov.hmrc.student_loans.thresholds.plan_2": {
            str(y): round(
                base_2026 * rpi_index(f"{y}-04-06") / rpi_2026
            )
            for y in range(2027, 2031)
        }
    }


def calculate_threshold_freeze_impact():
    """Calculate budgetary impact of cancelling the threshold freeze.

    Baseline = current law (frozen thresholds, PE-UK default).
    Reform = uprated thresholds.
    Impact = reformed − baseline (negative = cost to government).
    """
    print("Calculating uprated thresholds...")
    uprated_thresholds = calculate_uprated_thresholds()
    thresholds = uprated_thresholds[
        "gov.hmrc.student_loans.thresholds.plan_2"
    ]
    for year, value in thresholds.items():
        print(f"  {year}: £{value:,}")

    print("Creating baseline simulation (current law, frozen thresholds)...")
    baseline = Microsimulation(
        scenario=Scenario(
            simulation_modifier=connect_student_loan_variables,
        )
    )

    print("Creating reformed simulation (uprated thresholds)...")
    reformed = Microsimulation(
        scenario=Scenario(
            parameter_changes=uprated_thresholds,
            simulation_modifier=connect_student_loan_variables,
        )
    )

    results = []
    for year in YEARS:
        print(f"  Calculating year {year}...")
        b = baseline.calculate("student_loan_repayment", period=year)
        r = reformed.calculate("student_loan_repayment", period=year)
        impact = (r - b).sum() / 1e9
        results.append(
            {
                "year": year,
                "impact_bn": round(float(impact), 3),
                "baseline_total_bn": round(float(b.sum() / 1e9), 3),
                "reformed_total_bn": round(float(r.sum() / 1e9), 3),
            }
        )
        print(f"    Impact: £{impact:+.3f}bn")

    return results


# ─── Reform 2: Cap interest at RPI ───────────────────────────────────────────


def calculate_interest_cap_impact():
    """Calculate budgetary impact of capping interest at RPI.

    The fiscal cost = lost interest income on the student loan book.
    Interest accrual increases the government's loan asset value; capping
    at RPI wipes out the 0-3% margin above RPI.

    Impact = reformed_interest_accrual − baseline_interest_accrual (negative
    = cost to government, i.e. the government earns less interest).

    Baseline = current law (RPI + 0–3%, balance tracked).
    Reform = RPI only (additional_rate = 0, balance tracked).
    """
    interest_cap_params = {
        "gov.hmrc.student_loans.interest_rates.plan_2.additional_rate": {
            str(y): 0.0 for y in YEARS
        },
    }

    print("Creating baseline simulation (current interest, balance-aware)...")
    baseline = Microsimulation(
        scenario=Scenario(
            simulation_modifier=connect_and_track_balances,
        )
    )

    print("Creating reformed simulation (RPI cap, balance-aware)...")
    reformed = Microsimulation(
        scenario=Scenario(
            parameter_changes=interest_cap_params,
            simulation_modifier=connect_and_track_balances,
        )
    )

    # Diagnostics: estimated loan book

    init_bal = baseline.calculate(
        "student_loan_initial_balance", period=BASE_YEAR
    )
    plan = baseline.calculate("student_loan_plan", period=BASE_YEAR)
    weight = baseline.calculate("person_weight", period=BASE_YEAR)
    p2_mask = plan.values == "PLAN_2"
    p2_book = (init_bal.values[p2_mask] * weight.values[p2_mask]).sum() / 1e9
    p2_avg = np.average(
        init_bal.values[p2_mask], weights=weight.values[p2_mask]
    )
    p2_count = weight.values[p2_mask].sum() / 1e6
    print(
        f"  Estimated Plan 2 book: £{p2_book:.0f}bn "
        f"({p2_count:.1f}m borrowers, avg balance £{p2_avg:,.0f})"
    )

    results = []
    for year in YEARS:
        print(f"  Calculating year {year}...")
        # Interest accrual: the main fiscal impact
        b_interest = baseline.calculate(
            "student_loan_interest_accrual", period=year
        )
        r_interest = reformed.calculate(
            "student_loan_interest_accrual", period=year
        )
        interest_impact = (r_interest - b_interest).sum() / 1e9

        # Repayment change: secondary effect from balance tracking
        b_repay = baseline.calculate(
            "student_loan_repayment", period=year
        )
        r_repay = reformed.calculate(
            "student_loan_repayment", period=year
        )
        repayment_impact = (r_repay - b_repay).sum() / 1e9

        results.append(
            {
                "year": year,
                "interest_income_impact_bn": round(
                    float(interest_impact), 3
                ),
                "repayment_impact_bn": round(float(repayment_impact), 3),
                "baseline_interest_bn": round(
                    float(b_interest.sum() / 1e9), 3
                ),
                "reformed_interest_bn": round(
                    float(r_interest.sum() / 1e9), 3
                ),
            }
        )
        print(
            f"    Lost interest income: £{interest_impact:+.3f}bn"
            f"  |  Repayment change: £{repayment_impact:+.3f}bn"
        )

    return results


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Student Loan Reform: Budgetary Impact Calculator")
    print("=" * 60)

    # Reform 1: Cancel the threshold freeze
    print("\n--- Reform 1: Cancel the threshold freeze ---")
    threshold_results = calculate_threshold_freeze_impact()

    # Reform 2: Cap interest at RPI (balance-tracking microsim)
    print("\n--- Reform 2: Cap interest at RPI ---")
    interest_results = calculate_interest_cap_impact()

    output = {
        "cancel_threshold_freeze": {
            "description": (
                "Cancelling the Plan 2 student loan repayment threshold "
                "freeze. Baseline is current law (threshold frozen at "
                "£29,385 for 2027-2029). Reform uprates the threshold "
                "with RPI each year."
            ),
            "methodology": "PolicyEngine UK microsimulation",
            "annual_impacts": threshold_results,
        },
        "cap_interest_at_rpi": {
            "description": (
                "Capping Plan 2 student loan interest at RPI (removing "
                "the 0-3% margin above RPI). Fiscal cost = lost interest "
                "income on the student loan book. Interest accrual "
                "increases the government's loan asset value; capping "
                "at RPI wipes out the margin."
            ),
            "methodology": (
                "PolicyEngine UK microsimulation with balance tracking "
                "(cohort-specific initial balance: £45.6k for 2025 grads, "
                "deflated by RPI for earlier cohorts)"
            ),
            "annual_impacts": interest_results,
        },
    }

    with open("budget_impact.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nReform 1: Cancel the threshold freeze")
    print(
        f"{'Year':<8} {'Impact (£bn)':<15} "
        f"{'Baseline (£bn)':<16} {'Reformed (£bn)'}"
    )
    print("-" * 55)
    for r in threshold_results:
        print(
            f"{r['year']:<8} {r['impact_bn']:>+12.3f}   "
            f"{r['baseline_total_bn']:>12.3f}   "
            f"{r['reformed_total_bn']:>12.3f}"
        )

    print("\nReform 2: Cap interest at RPI (lost interest income on loan book)")
    print(
        f"{'Year':<8} {'Interest (£bn)':<16} "
        f"{'Baseline (£bn)':<16} {'Reformed (£bn)':<16} "
        f"{'Repay Δ (£bn)'}"
    )
    print("-" * 75)
    for r in interest_results:
        print(
            f"{r['year']:<8} {r['interest_income_impact_bn']:>+13.3f}   "
            f"{r['baseline_interest_bn']:>12.3f}   "
            f"{r['reformed_interest_bn']:>12.3f}   "
            f"{r['repayment_impact_bn']:>+11.3f}"
        )

    print("\nOutput written to budget_impact.json")


if __name__ == "__main__":
    main()
