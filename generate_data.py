import numpy as np
import pandas as pd
import os

RNG = np.random.RandomState(42)

def generate(n=3000):
    loan_amnt = RNG.normal(12000, 5000, n).clip(1000, 50000).astype(int)
    annual_inc = (RNG.normal(50000, 20000, n).clip(8000, 300000)).astype(int)
    age = RNG.normal(40, 11, n).clip(18, 75).astype(int)
    term = RNG.choice(['36 months', '60 months'], size=n, p=[0.8, 0.2])
    purpose = RNG.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'other'], size=n)
    home_ownership = RNG.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], size=n, p=[0.45,0.25,0.25,0.05])
    emp_length = RNG.choice(['<1','1-3','3-5','5-10','10+'], size=n, p=[0.05,0.25,0.3,0.25,0.15])
    delinq_2yrs = RNG.poisson(0.3, n).clip(0, 10)
    revolving_util = RNG.normal(40, 25, n).clip(0, 150)
    open_acc = RNG.poisson(6, n).clip(1, 30)

    risk_score = (
        0.00005 * loan_amnt
        + 0.00001 * (150000 - annual_inc)  
        + 0.02 * (delinq_2yrs)
        + 0.01 * np.maximum(0, revolving_util - 60)
        - 0.01 * (open_acc - 5)
        + 0.02 * (age < 25)
        + 0.01 * (term == '60 months').astype(int)
    )

    prob_default = 1 / (1 + np.exp(-risk_score + RNG.normal(0,0.5,n)))
    prob_default = (prob_default - prob_default.min()) / (prob_default.max() - prob_default.min())
    prob_default = prob_default * 0.58 + 0.02

    default = (RNG.rand(n) < prob_default).astype(int)

    df = pd.DataFrame({
        'loan_amnt': loan_amnt,
        'annual_inc': annual_inc,
        'age': age,
        'term': term,
        'purpose': purpose,
        'home_ownership': home_ownership,
        'emp_length': emp_length,
        'delinq_2yrs': delinq_2yrs,
        'revolving_util': np.round(revolving_util,2),
        'open_acc': open_acc,
        'default': default
    })

    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate(3000)
    df.to_csv('data/credit_data.csv', index=False)
    print("Saved data/credit_data.csv with shape", df.shape)
