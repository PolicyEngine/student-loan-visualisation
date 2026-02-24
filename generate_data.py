"""
Student loan repayment model: threshold uprating vs interest rate caps.

Models Plan 2 (post-2012) student loans across the income distribution,
comparing the distributional impact of:
1. Cancelling the threshold freeze (uprate with earnings)
2. Capping the interest rate at RPI (analogous to social housing CPI+1%)

Key mechanics:
- Repayment: 9% of gross income above the threshold
- Interest: RPI + 0-3% (scaled by income between threshold and upper limit)
- Loan written off after 30 years
- Average graduate debt ~£45,000

Baseline: threshold frozen at £29,385 from 2027 (Autumn Budget 2025 policy).
Economic assumptions: OBR November 2025 Economic and Fiscal Outlook.
"""

import json

# Current Plan 2 parameters (2026-27)
# Threshold rises to £29,385 in April 2026, then frozen for 3 years (2027-30)
CURRENT_THRESHOLD = 29_385
FREEZE_YEARS = 3  # frozen for 2027-28, 2028-29, 2029-30
REPAYMENT_RATE = 0.09
MAX_INTEREST_MARGIN = 0.03  # up to 3% above RPI
UPPER_INCOME_FOR_INTEREST = 51_245  # 2026-27 value
LOAN_TERM_YEARS = 30
AVERAGE_DEBT = 45_000

# Income distribution: £15k to £120k
INCOMES = list(range(0, 120_001, 1_000))

# OBR November 2025 EFO forecasts (year-by-year, starting from 2026-27)
# Source: https://obr.uk/efo/economic-and-fiscal-outlook-november-2025/
OBR_RPI_FORECASTS = {
    0: 0.037,  # 2026-27
    1: 0.031,  # 2027-28
    2: 0.029,  # 2028-29
    3: 0.029,  # 2029-30
    4: 0.023,  # 2030-31
}
OBR_RPI_LONG_RUN = 0.029

OBR_EARNINGS_GROWTH = {
    0: 0.032,  # 2026-27
    1: 0.021,  # 2027-28
    2: 0.021,  # 2028-29
    3: 0.022,  # 2029-30
    4: 0.023,  # 2030-31
}
OBR_EARNINGS_LONG_RUN = 0.023


def get_rpi(year: int) -> float:
    return OBR_RPI_FORECASTS.get(year, OBR_RPI_LONG_RUN)


def get_earnings_growth(year: int) -> float:
    return OBR_EARNINGS_GROWTH.get(year, OBR_EARNINGS_LONG_RUN)


def interest_rate(
    income: float, threshold: float, rpi: float, max_margin: float
) -> float:
    """Calculate Plan 2 interest rate based on income."""
    if income <= threshold:
        return rpi
    elif income >= UPPER_INCOME_FOR_INTEREST:
        return rpi + max_margin
    else:
        proportion = (income - threshold) / (
            UPPER_INCOME_FOR_INTEREST - threshold
        )
        return rpi + max_margin * proportion


def get_threshold_schedule(mode: str) -> dict:
    """Return a year-by-year threshold schedule.

    Year 0 = 2026-27 (threshold at £29,385).
    The Autumn Budget 2025 froze the threshold for 3 years from 2027-28,
    so years 1-3 are frozen, and uprating resumes from year 4 (2030-31).

    mode:
      'frozen' - frozen for 3 years then uprated (current policy)
      'uprated' - uprated with OBR earnings growth every year (cancel the freeze)
    """
    thresholds = {}
    t = CURRENT_THRESHOLD
    for year in range(LOAN_TERM_YEARS):
        thresholds[year] = t
        if mode == "uprated":
            # Cancel the freeze: uprate every year
            t *= 1 + get_earnings_growth(year)
        elif mode == "frozen":
            # Baseline: freeze for years 1-3, then resume uprating
            if year >= FREEZE_YEARS:
                t *= 1 + get_earnings_growth(year)
            # else: t stays the same (frozen)
    return thresholds


def simulate_repayment(
    starting_income: float,
    debt: float = AVERAGE_DEBT,
    threshold_mode: str = "frozen",
    max_interest_margin: float = MAX_INTEREST_MARGIN,
    years: int = LOAN_TERM_YEARS,
) -> dict:
    """Simulate lifetime student loan repayment for a given starting salary."""
    balance = debt
    total_repaid = 0
    annual_repayments = []
    income = starting_income

    thresholds = get_threshold_schedule(threshold_mode)

    for year in range(years):
        rpi = get_rpi(year)
        earnings_growth = get_earnings_growth(year)
        threshold = thresholds[year]

        # Annual repayment
        repayment = max(0, (income - threshold) * REPAYMENT_RATE)
        repayment = min(repayment, balance)
        balance -= repayment
        total_repaid += repayment
        annual_repayments.append(repayment)

        # Interest accrues on remaining balance
        rate = interest_rate(income, threshold, rpi, max_interest_margin)
        balance *= 1 + rate
        balance = max(0, balance)

        # Nominal earnings growth (OBR forecasts)
        income *= 1 + earnings_growth

        if balance <= 0:
            break

    written_off = max(0, balance)

    return {
        "total_repaid": round(total_repaid),
        "written_off": round(written_off),
        "annual_repayment_year1": round(annual_repayments[0])
        if annual_repayments
        else 0,
        "years_repaying": len([r for r in annual_repayments if r > 0]),
        "effective_rate": round(total_repaid / debt * 100, 1) if debt > 0 else 0,
    }


def simulate_yearly(
    starting_income: float,
    debt: float = AVERAGE_DEBT,
    threshold_mode: str = "frozen",
    max_interest_margin: float = MAX_INTEREST_MARGIN,
    max_years: int = 40,
) -> list:
    """Simulate year-by-year loan repayment. Returns list of annual repayments."""
    balance = debt
    income = starting_income
    thresholds = get_threshold_schedule(threshold_mode)
    yearly = []

    for year in range(max_years):
        if year >= LOAN_TERM_YEARS or balance <= 0:
            yearly.append({"year": year + 1, "annual_repayment": 0, "balance": 0})
            income *= 1 + get_earnings_growth(year)
            continue

        rpi = get_rpi(year)
        threshold = thresholds[year]

        repayment = max(0, (income - threshold) * REPAYMENT_RATE)
        repayment = min(repayment, balance)
        balance -= repayment

        rate = interest_rate(income, threshold, rpi, max_interest_margin)
        balance *= 1 + rate
        balance = max(0, balance)

        yearly.append(
            {"year": year + 1, "annual_repayment": repayment, "balance": balance}
        )
        income *= 1 + get_earnings_growth(year)

    return yearly


def generate_scenarios() -> dict:
    """Generate data for all policy scenarios across the income distribution."""

    scenarios = {
        "baseline": {
            "label": "Current system (threshold frozen)",
            "description": (
                f"Threshold frozen at \u00a3{CURRENT_THRESHOLD:,.0f} "
                f"from 2027, interest RPI + 0\u20133%"
            ),
            "threshold_mode": "frozen",
            "max_interest_margin": MAX_INTEREST_MARGIN,
        },
        "cancel_freeze": {
            "label": "Cancel the threshold freeze",
            "description": (
                "Uprate threshold annually with average earnings growth (OBR forecasts)"
            ),
            "threshold_mode": "uprated",
            "max_interest_margin": MAX_INTEREST_MARGIN,
        },
        "interest_cap_rpi": {
            "label": "Cap interest at RPI",
            "description": (
                "Cap the interest rate at RPI (0% margin), "
                "as proposed by the Conservative Party"
            ),
            "threshold_mode": "frozen",
            "max_interest_margin": 0.0,
        },
    }

    results = {}

    for scenario_key, scenario in scenarios.items():
        scenario_results = []
        for income in INCOMES:
            sim = simulate_repayment(
                starting_income=income,
                threshold_mode=scenario["threshold_mode"],
                max_interest_margin=scenario["max_interest_margin"],
            )
            scenario_results.append(
                {
                    "income": income,
                    "total_repaid": sim["total_repaid"],
                    "written_off": sim["written_off"],
                    "annual_repayment_year1": sim["annual_repayment_year1"],
                    "years_repaying": sim["years_repaying"],
                    "effective_rate": sim["effective_rate"],
                }
            )

        results[scenario_key] = {
            "label": scenario["label"],
            "description": scenario["description"],
            "data": scenario_results,
        }

    # Compute savings vs baseline
    baseline_data = {d["income"]: d for d in results["baseline"]["data"]}

    for scenario_key in scenarios:
        if scenario_key == "baseline":
            continue
        for d in results[scenario_key]["data"]:
            baseline = baseline_data[d["income"]]
            d["saving_vs_baseline"] = baseline["total_repaid"] - d["total_repaid"]
            d["saving_pct"] = (
                round(
                    d["saving_vs_baseline"] / baseline["total_repaid"] * 100, 1
                )
                if baseline["total_repaid"] > 0
                else 0
            )

    # Graduate salary percentiles (approximate, ONS-based)
    percentiles = {
        "10th": 22_000,
        "25th": 27_000,
        "50th": 35_000,
        "75th": 50_000,
        "90th": 70_000,
    }

    # Threshold schedule for reference
    uprated_schedule = get_threshold_schedule("uprated")
    threshold_schedule = {
        f"year_{y}": round(uprated_schedule[y]) for y in range(6)
    }

    # Annual repayment profiles for selected salary levels
    profile_salaries = [25_000, 35_000, 50_000, 70_000]
    annual_profiles = {}
    for sal in profile_salaries:
        profiles = {}
        for scenario_key, scenario in scenarios.items():
            if scenario_key == "baseline":
                continue
            yearly = simulate_yearly(
                starting_income=sal,
                threshold_mode=scenario["threshold_mode"],
                max_interest_margin=scenario["max_interest_margin"],
            )
            profiles[scenario_key] = [
                round(r["annual_repayment"]) for r in yearly
            ]
        annual_profiles[str(sal)] = profiles

    output = {
        "parameters": {
            "current_threshold": CURRENT_THRESHOLD,
            "repayment_rate": REPAYMENT_RATE,
            "max_interest_margin": MAX_INTEREST_MARGIN,
            "upper_income_for_interest": UPPER_INCOME_FOR_INTEREST,
            "loan_term_years": LOAN_TERM_YEARS,
            "average_debt": AVERAGE_DEBT,
            "source": "OBR November 2025 Economic and Fiscal Outlook",
        },
        "obr_assumptions": {
            "rpi_forecasts": {
                "2026-27": OBR_RPI_FORECASTS[0],
                "2027-28": OBR_RPI_FORECASTS[1],
                "2028-29": OBR_RPI_FORECASTS[2],
                "2029-30": OBR_RPI_FORECASTS[3],
                "2030-31": OBR_RPI_FORECASTS[4],
                "long_run": OBR_RPI_LONG_RUN,
            },
            "earnings_growth_forecasts": {
                "2026-27": OBR_EARNINGS_GROWTH[0],
                "2027-28": OBR_EARNINGS_GROWTH[1],
                "2028-29": OBR_EARNINGS_GROWTH[2],
                "2029-30": OBR_EARNINGS_GROWTH[3],
                "2030-31": OBR_EARNINGS_GROWTH[4],
                "long_run": OBR_EARNINGS_LONG_RUN,
            },
        },
        "threshold_schedule_if_uprated": threshold_schedule,
        "scenarios": results,
        "graduate_income_percentiles": percentiles,
        "annual_profiles": annual_profiles,
    }

    return output


if __name__ == "__main__":
    data = generate_scenarios()
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(
        f"Generated data for {len(data['scenarios'])} scenarios "
        f"across {len(INCOMES)} income levels"
    )
    print("Output written to data.json")

    # Print key stats
    scenarios = data["scenarios"]
    baseline = {d["income"]: d for d in scenarios["baseline"]["data"]}
    cancel = {d["income"]: d for d in scenarios["cancel_freeze"]["data"]}
    cap = {d["income"]: d for d in scenarios["interest_cap_rpi"]["data"]}

    for inc in [35_000, 50_000, 70_000]:
        b = baseline[inc]["total_repaid"]
        c = cancel[inc]["total_repaid"]
        i = cap[inc]["total_repaid"]
        print(
            f"\n£{inc//1000}k salary: "
            f"baseline £{b:,}, "
            f"cancel freeze saves £{b - c:,}, "
            f"RPI cap saves £{b - i:,}"
        )
