# Dataset

This project uses the AI4I 2020 Predictive Maintenance dataset. It is a
**synthetic dataset** created to illustrate predictive-maintenance methods;
results from it do not establish performance on real industrial equipment.

The original dataset is available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
Download `ai4i2020.csv` and place it at `dataset/ai4i2020.csv` in the project
root. This workspace already contains a local copy at that location. The CSV
is intentionally excluded from Git to keep the repository small.

Alternatively, point the loader to another copy without changing code:

```powershell
$env:PREDICTIVE_MAINTENANCE_DATA = "C:\path\to\ai4i2020.csv"
```

The expected target is `Machine failure`. See the project README and notebook
for the schema, class balance, and the rationale for excluding the failure
mode columns from model features.
