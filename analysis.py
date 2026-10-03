"""
Sales Data Analysis | تحليل بيانات المبيعات
Generates a realistic dataset, cleans it, analyses it and builds a dashboard.
Run: python analysis.py
Outputs: sales_data.csv, dashboard.png, report.txt
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1. Create dataset ----------
n = 3000
products = {"Laptop": 900, "Phone": 600, "Headphones": 80,
            "Tablet": 400, "Smart Watch": 200}
regions = ["Cairo", "Alexandria", "Giza", "Mansoura", "Aswan"]
df = pd.DataFrame({
    "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
    "product": rng.choice(list(products), n, p=[.15, .25, .3, .1, .2]),
    "region": rng.choice(regions, n, p=[.35, .2, .2, .15, .1]),
    "quantity": rng.integers(1, 6, n),
})
df["unit_price"] = df["product"].map(products) * rng.uniform(.9, 1.1, n)
df.loc[rng.choice(n, 40, replace=False), "region"] = np.nan      # dirty data
df.to_csv("sales_data.csv", index=False)

# ---------- 2. Clean ----------
df = pd.read_csv("sales_data.csv", parse_dates=["date"])
df["region"] = df["region"].fillna("Unknown")
df["revenue"] = df["quantity"] * df["unit_price"]
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

# ---------- 3. Analyse ----------
total = df["revenue"].sum()
monthly = df.groupby("month")["revenue"].sum()
by_prod = df.groupby("product")["revenue"].sum().sort_values()
by_reg = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
growth = monthly.pct_change().mean() * 100
best_month = monthly.idxmax().strftime("%B")

# ---------- 4. Dashboard ----------
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(14, 8.5))
fig.suptitle("Sales Dashboard 2025", fontsize=20, fontweight="bold", y=.98)

kpis = [("Total Revenue", f"${total/1e6:.2f}M"),
        ("Orders", f"{len(df):,}"),
        ("Avg Order", f"${df['revenue'].mean():,.0f}"),
        ("Best Month", best_month)]
for i, (t, v) in enumerate(kpis):
    ax = fig.add_axes([.04 + i*.24, .84, .22, .09]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#1f3b73", transform=ax.transAxes))
    ax.text(.5, .62, v, ha="center", va="center", color="white", fontsize=18, fontweight="bold")
    ax.text(.5, .22, t, ha="center", va="center", color="#cfd8ff", fontsize=10)

ax1 = fig.add_axes([.06, .45, .55, .3])
ax1.plot(monthly.index, monthly.values / 1e3, marker="o", color="#1f3b73", lw=2.5)
ax1.fill_between(monthly.index, monthly.values / 1e3, alpha=.15, color="#1f3b73")
ax1.set_title("Monthly Revenue (K$)", loc="left", fontweight="bold")

ax2 = fig.add_axes([.68, .45, .29, .3])
ax2.pie(by_reg, labels=by_reg.index, autopct="%1.0f%%", startangle=90,
        colors=plt.cm.Blues(np.linspace(.9, .35, len(by_reg))))
ax2.set_title("Revenue by Region", fontweight="bold")

ax3 = fig.add_axes([.12, .07, .38, .28])
ax3.barh(by_prod.index, by_prod.values / 1e3, color="#2a9d8f")
ax3.set_title("Revenue by Product (K$)", loc="left", fontweight="bold")

ax4 = fig.add_axes([.58, .07, .30, .28])
pivot = df.pivot_table(index="region", columns="product", values="revenue", aggfunc="sum") / 1e3
im = ax4.imshow(pivot, cmap="YlGnBu", aspect="auto")
ax4.set_xticks(range(len(pivot.columns))); ax4.set_xticklabels(pivot.columns, rotation=30, fontsize=8)
ax4.set_yticks(range(len(pivot.index))); ax4.set_yticklabels(pivot.index, fontsize=8)
ax4.set_title("Region x Product Heatmap (K$)", fontweight="bold"); ax4.grid(False)
plt.colorbar(im, ax=ax4, fraction=.05)
fig.savefig("dashboard.png", dpi=130)

# ---------- 5. Report ----------
report = f"""SALES ANALYSIS REPORT 2025
==========================
Total revenue : ${total:,.0f}
Orders        : {len(df):,}
Avg order     : ${df['revenue'].mean():,.2f}
Best month    : {best_month}
Avg monthly growth : {growth:.2f}%
Top product   : {by_prod.index[-1]} (${by_prod.iloc[-1]:,.0f})
Top region    : {by_reg.index[0]} (${by_reg.iloc[0]:,.0f})
Missing regions cleaned : 40 rows -> 'Unknown'
"""
open("report.txt", "w").write(report)
print(report)
