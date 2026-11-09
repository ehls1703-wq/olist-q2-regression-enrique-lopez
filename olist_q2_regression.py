
import kagglehub
import pandas as pd
from pathlib import Path

# Download or locate the public Olist dataset
data_path = Path(
    kagglehub.dataset_download("olistbr/brazilian-ecommerce")
)

# Read the four datasets for Q2
items = pd.read_csv(data_path / "olist_order_items_dataset.csv")
products = pd.read_csv(data_path / "olist_products_dataset.csv")
orders = pd.read_csv(data_path / "olist_orders_dataset.csv")
customers = pd.read_csv(data_path / "olist_customers_dataset.csv")

# Check dataset sizes
print("Order items:", len(items))
print("Products:", len(products))
print("Orders:", len(orders))
print("Customers:", len(customers))

# Check the available columns
print("\nOrder items columns:", items.columns.tolist())
print("\nProducts columns:", products.columns.tolist())
print("\nOrders columns:", orders.columns.tolist())
print("\nCustomers columns:", customers.columns.tolist())

# Q2: Check columns needed for the data joins
required_columns = {
    "Order items": (
        items,
        ["order_id", "product_id", "freight_value"]
    ),
    "Products": (
        products,
        [
            "product_id",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ),
    "Orders": (
        orders,
        ["order_id", "customer_id"]
    ),
    "Customers": (
        customers,
        ["customer_id", "customer_state"]
    ),
}

for name, (dataset, columns) in required_columns.items():
    missing = [col for col in columns if col not in dataset.columns]
    if missing:
        print(f"{name}: MISSING {missing}")
    else:
        print(f"{name}: OK")

# Q2: Join the Olist datasets
df = items.merge(
    products,
    on="product_id",
    how="inner",
    validate="many_to_one"
)

df = df.merge(
    orders[["order_id", "customer_id"]],
    on="order_id",
    how="inner",
    validate="many_to_one"
)

df = df.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="inner",
    validate="many_to_one"
)

# Select the variables needed for Q2
columns = [
    "freight_value",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "customer_state"
]

df = df[columns].copy()

print("\n--- Q2 DATA CHECK ---")
print("Rows after joining:", len(df))
print("\nMissing values:")
print(df.isna().sum())

# Convert grams to kilograms
df["weight_kg"] = df["product_weight_g"] / 1000

print("\nFirst five rows:")
print(df.head().to_string(index=False))

# Q2: Check missing and invalid values
print("\n--- DATA CLEANING ---")

print("Rows before cleaning:", len(df))

model_columns = [
    "freight_value",
    "weight_kg",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "customer_state"
]

clean_df = df[model_columns].dropna().copy()

print("Rows after removing missing values:", len(clean_df))

# Remove invalid negative values and nonpositive dimensions
clean_df = clean_df[
    (clean_df["freight_value"] >= 0) &
    (clean_df["weight_kg"] > 0) &
    (clean_df["product_length_cm"] > 0) &
    (clean_df["product_height_cm"] > 0) &
    (clean_df["product_width_cm"] > 0)
].copy()

print("Rows after validity checks:", len(clean_df))
print("Rows removed in total:", len(df) - len(clean_df))

print("\nCustomer states:", clean_df["customer_state"].nunique())
print("\nClean data preview:")
print(clean_df.head().to_string(index=False))

# Q2: Fit linear regression
import statsmodels.formula.api as smf

formula = (
    "freight_value ~ weight_kg "
    "+ product_length_cm "
    "+ product_height_cm "
    "+ product_width_cm "
    "+ C(customer_state, Treatment(reference='SP'))"
)

model = smf.ols(
    formula=formula,
    data=clean_df
).fit(cov_type="HC3")

print("\n--- LINEAR REGRESSION ---")
print("Observations:", int(model.nobs))

# Print estimates, standard errors and confidence intervals
results = pd.DataFrame({
    "Estimate": model.params,
    "Std. Error": model.bse,
    "CI Lower": model.conf_int()[0],
    "CI Upper": model.conf_int()[1]
})

print(results.round(3).to_string())
print("\nR-squared:", round(model.rsquared, 3))

# Export regression coefficients for the memo
results.to_csv(
    "q2_regression_results.csv",
    index_label="Predictor"
)

print("\nRegression results saved successfully!")

# Q2: Execute SQL and fit the order-level regression
import duckdb

con = duckdb.connect()

con.register("items", items)
con.register("products", products)
con.register("orders", orders)
con.register("customers", customers)

sql = Path("olist_q2_query.sql").read_text()
order_df = con.execute(sql).df()

print("\n--- ORDER LEVEL DATA ---")
print("Orders before cleaning:", len(order_df))

order_df = order_df.dropna().copy()
order_df = order_df[
    (order_df["freight_value"] >= 0) &
    (order_df["weight_kg"] > 0) &
    (order_df["product_length_cm"] > 0) &
    (order_df["product_height_cm"] > 0) &
    (order_df["product_width_cm"] > 0)
].copy()

print("Orders after cleaning:", len(order_df))

order_model = smf.ols(
    formula=formula,
    data=order_df
).fit(cov_type="HC3")

print("Observations:", int(order_model.nobs))
print("R-squared:", round(order_model.rsquared, 3))

order_results = pd.DataFrame({
    "Estimate": order_model.params,
    "Std. Error": order_model.bse,
    "CI Lower": order_model.conf_int()[0],
    "CI Upper": order_model.conf_int()[1]
})

order_results.to_csv(
    "q2_order_results.csv",
    index_label="Predictor"
)

print("\nOrder-level regression saved successfully!")








