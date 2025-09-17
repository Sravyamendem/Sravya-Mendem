# CS5710 Machine Learning – Homework 1

## Student Info
Name : Sravya Mendem
Srudent ID : 700765926
---

## Contents

This repository contains my solutions for **Homework 1** in CS5710 Machine Learning.

### Files included
- `homework1_solutions.pdf` – Written solutions with plots included
- `homework1_solutions.ipynb` – Jupyter Notebook with step-by-step answers and runnable code
- `linear_regression_gd_vs_closed_form.py` – Python script for Question 7 (closed-form vs gradient descent)

---

## Questions Overview

1. **Function Approximation by Hand**  
   Computed predictions, residuals, squared errors, and MSE for two candidate models.

2. **Random Guessing Practice**  
   Evaluated cost function for given points, showed inefficiency of random guessing.

3. **First Gradient Descent Iteration**  
   Manually computed gradient, updated parameters, and compared cost before/after update.

4. **Random Guessing vs Gradient Descent**  
   Compared error values from random guesses and the first GD step.

5. **Recognizing Underfitting and Overfitting**  
   Identified underfitting scenario and suggested fixes.

6. **Comparing Models**  
   Discussed overfitting (Model A) vs underfitting (Model B), bias–variance tradeoff, and improvements.

7. **Programming Problem – Gradient Descent for Linear Regression**  
   - Generated synthetic dataset `y=3+4x+ϵ`  
   - Solved with **Normal Equation (Closed-form)**  
   - Implemented **Gradient Descent** from scratch  
   - Plotted raw data, fitted lines, and GD loss curve  
   - Verified GD converges to same solution as closed-form

---

## Results (Q7)
- Closed-form: intercept ≈ 2.69, slope ≈ 4.13  
- Gradient Descent: intercept ≈ 2.69, slope ≈ 4.13  
- Both methods agree, confirming convergence.

---

## How to Run the Code

### Option 1 – Python Script
```bash
python linear_regression_gd_vs_closed_form.py
```

### Option 2 – Jupyter Notebook
```bash
jupyter notebook homework1_solutions.ipynb
```

---

## Notes
- All code is implemented **from scratch** (no scikit-learn for linear regression).  
- Plots included in both the PDF and Jupyter notebook.  

"# Sravya-Mendem" 
