# Student Loan Reform Visualisation

Interactive visualisation comparing the distributional impact of two proposed UK student loan reforms across the income distribution.

**Live site:** https://policyengine.org/uk/student-loan-visualisation

## Reforms modelled

- **Cancel the threshold freeze** — uprate the repayment threshold with earnings growth each year, as advocated by Martin Lewis
- **Cap interest at RPI** — eliminate the interest margin above RPI, as proposed by Kemi Badenoch

## Data

The model uses Plan 2 (post-2012) loan parameters with economic assumptions from the [OBR November 2025 Economic and Fiscal Outlook](https://obr.uk/efo/economic-and-fiscal-outlook-november-2025/). Run `python generate_data.py` to regenerate `data.json`.

## Deployment

Deployed on Vercel. Merges to `main` trigger automatic production deployments.
