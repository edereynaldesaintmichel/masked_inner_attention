import json
import torch

FIELDS_AND_LIMITS = {
    "revenue": 1e12, # Apple's revenue is ~400B
    "costOfRevenue": 8e11,
    "grossProfit": 4e11,
    "grossProfitRatio": 1.0, # As a ratio, should be between 0 and 1
    "researchAndDevelopmentExpenses": 5e10,
    "generalAndAdministrativeExpenses": 5e10,
    "sellingAndMarketingExpenses": 5e10,
    "sellingGeneralAndAdministrativeExpenses": 1e11,
    "otherExpenses": 5e10,
    "operatingExpenses": 2e11,
    "costAndExpenses": 9e11,
    "interestIncome": 5e10,
    "interestExpense": 5e10,
    "depreciationAndAmortization": 5e10,
    "ebitda": 4e11,
    "ebitdaratio": 1.0,
    "operatingIncome": 3e11,
    "operatingIncomeRatio": 1.0,
    "totalOtherIncomeExpensesNet": 5e10,
    "incomeBeforeTax": 3e11,
    "incomeBeforeTaxRatio": 1.0,
    "incomeTaxExpense": 1e11,
    "netIncome": 2e11,
    "netIncomeRatio": 1.0,
    "eps": 1000, # Very generous
    "epsdiluted": 1000,
    "weightedAverageShsOut": 2e10, # Apple has about 16B
    "weightedAverageShsOutDil": 2e10,
    "cashAndCashEquivalents": 3e11,
    "shortTermInvestments": 3e11,
    "cashAndShortTermInvestments": 4e11,
    "netReceivables": 2e11,
    "inventory": 2e11,
    "otherCurrentAssets": 2e11,
    "totalCurrentAssets": 5e11,
    "propertyPlantEquipmentNet": 5e11,
    "goodwill": 4e11,
    "intangibleAssets": 4e11,
    "goodwillAndIntangibleAssets": 5e11,
    "longTermInvestments": 5e11,
    "taxAssets": 1e11,
    "otherNonCurrentAssets": 3e11,
    "totalNonCurrentAssets": 2e12,
    "otherAssets": 3e11,
    "totalAssets": 3e12,
    "accountPayables": 2e11,
    "shortTermDebt": 3e11,
    "taxPayables": 1e11,
    "deferredRevenue": 1e11,
    "otherCurrentLiabilities": 2e11,
    "totalCurrentLiabilities": 5e11,
    "longTermDebt": 5e11,
    "deferredRevenueNonCurrent": 1e11,
    "deferredTaxLiabilitiesNonCurrent": 1e11,
    "otherNonCurrentLiabilities": 2e11,
    "totalNonCurrentLiabilities": 1e12,
    "otherLiabilities": 2e11,
    "capitalLeaseObligations": 2e11,
    "totalLiabilities": 2e12,
    "preferredStock": 1e11,
    "commonStock": 1e11,
    "retainedEarnings": 5e11,
    "accumulatedOtherComprehensiveIncomeLoss": 1e11,
    "othertotalStockholdersEquity": 2e11,
    "totalStockholdersEquity": 1e12,
    "totalEquity": 1e12,
    "totalLiabilitiesAndStockholdersEquity": 3e12,
    "minorityInterest": 2e11,
    "totalLiabilitiesAndTotalEquity": 3e12,
    "totalInvestments": 1e12,
    "totalDebt": 1e12,
    "netDebt": 1e12,
    "calendarYear": None,
    "reportedCurrency": None # This should be a string
}

def niceify_data():
    financial_statements = {}
    for i in range(26):
        with open(f'data/full_reports_{i}.json', 'r+') as file:
            to_add = json.load(file)
            financial_statements = {**financial_statements, **to_add}

    # Count field occurrences
    all_fields_counter = {}
    # financial_statements = dict(islice(financial_statements.items(), 5))
    # fields = [key for key, value in next(iter(financial_statements.values()))[0].items() if isinstance(value, (int, float))] + ["calendarYear", "reportedCurrency"]

    data = []
    currency_indices = {"USD": 0, "EUR": 1, "CAD": 2, "CNY": 3, "IDR": 4, "AUD": 5, "ILS": 6, "GBP": 7, "DKK": 8, "BRL": 9, "NOK": 10, "PHP": 11, "SEK": 12, "TWD": 13, "CHF": 14, "TRY": 15, "NZD": 16, "SGD": 17, "JPY": 18, "HKD": 19, "NGN": 20, "ZAR": 21, "PEN": 22, "MYR": 23, "THB": 24, "CLP": 26, "PLN": 27, "MXN": 28, "NIS": 29, "SAR": 30, "PGK": 31, "COP": 32, "INR": 33, "ARS": 34, "GEL": 35, "GHS": 36, "CZK": 37, "EGP": 38, "RON": 39, "HUF": 40, "RUB": 41, "KRW": 42, "KZT": 43, "NAD": 44, "VND": 45}
    currency_exchange_rates = {"USD": 1.0, "EUR": 0.93, "CAD": 1.37, "CNY": 7.24, "IDR": 15865.0, "AUD": 1.52, "ILS": 3.74, "GBP": 0.8, "DKK": 6.97, "BRL": 4.95, "NOK": 10.85, "PHP": 57.68, "SEK": 10.57, "TWD": 32.45, "CHF": 0.91, "TRY": 32.24, "NZD": 1.65, "SGD": 1.35, "JPY": 151.64, "HKD": 7.82, "NGN": 1487.96, "ZAR": 18.96, "PEN": 3.72, "MYR": 4.77, "THB": 36.26, "CLP": 971.46, "PLN": 4.02, "MXN": 17.06, "NIS": 3.74, "SAR": 3.75, "PGK": 3.8, "COP": 3918.96, "INR": 83.5, "ARS": 879.65, "GEL": 2.68, "GHS": 13.89, "CZK": 23.47, "EGP": 47.6, "RON": 4.64, "HUF": 366.1, "RUB": 91.62, "KRW": 1334.42, "KZT": 450.82, "NAD": 18.96, "VND": 24535.0}
    all_financial_statements = []
    ratio_fields = ['grossProfitRatio', 'ebitdaratio', 'operatingIncomeRatio', 'incomeBeforeTaxRatio', 'netIncomeRatio']
    to_not_scale_with_exchange_rate = set(['calendarYear', 'reportedCurrency'] + ratio_fields)
    invalid_counter = 0
    general_counter = 0
    for company_statements in financial_statements.values():
        general_counter += 1
        list_statements = []
        is_valid = True
        for statement in company_statements:
            if float(statement['calendarYear']) < 2000:
                break
            currency = statement['reportedCurrency']
            if currency not in currency_indices:
                continue
            statement['reportedCurrency'] = currency_indices[currency]
            vector = []
            for field, limit in FIELDS_AND_LIMITS.items():
                value = statement[field]
                try:
                    value = float(value)
                except:
                    value = 0.0
                if field not in to_not_scale_with_exchange_rate:
                    value = value / currency_exchange_rates[currency]
                
                if limit is not None and abs(value) > limit:
                    is_valid = False
                    invalid_counter += 1
                    break

                vector.append(value)
            if not is_valid:
                break
            list_statements.append(vector)
        if not is_valid:
            continue
        all_financial_statements += list_statements
        list_statements = torch.tensor(list_statements)
        data.append(list_statements)

    all_financial_statements = torch.tensor(all_financial_statements)
    all_financial_statements = torch.nan_to_num(all_financial_statements, nan=0.0, posinf=0.0, neginf=0.0)
    std, mean = torch.std_mean(all_financial_statements[:,:-1], dim=0)
    std = torch.cat((std, torch.tensor(1).unsqueeze(0)), -1)
    mean = torch.cat((mean, torch.tensor(0).unsqueeze(0)), -1)

    normalized_data = []
    for reports in data:
        if reports.shape[0] == 0:
            continue
        reports = (reports - mean) / std
        normalized_data.append(reports)

    print(f'Share of invalid data: {invalid_counter/general_counter}')
    torch.save(std, 'std.pt')
    torch.save(mean, 'mean.pt')
    torch.save(normalized_data, 'full_data.pt')
    torch.save(currency_indices, 'currency_indices.pt')
    torch.save(currency_exchange_rates, 'currency_exchange_rates.pt')

    return normalized_data


niceify_data()