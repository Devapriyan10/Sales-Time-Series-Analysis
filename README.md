# Sales Time Series Analysis

## Project Description
This project performs time series analysis on sales data to uncover trends, seasonality, and residual patterns. By resampling sales data into hourly, daily, weekly, and monthly frequencies, the project helps identify high and low sales periods and provides actionable business insights. The analysis results in visual plots and reports to support data-driven decision-making for optimizing sales operations.

## How to Use the Code

### Prerequisites
Make sure to install the following Python libraries:
```bash
pip install pandas numpy matplotlib statsmodels
```

### Running the Code
1. **Prepare your sales data**: The CSV file should have:
   - `timestamp`: Date and time in the format `DD-MM-YYYY HH:MM`
   - `sales`: Numeric sales figures

2. **Run the analysis**:
   - Place your CSV file in the project directory.
   - Update the file path in the script (`sales_analysis.py`).
   - Run the script using:
     ```bash
     python sales_analysis.py
     ```

3. **Output**: The code generates:
   - Decomposed time series plots for observed, trend, seasonal, and residual components (saved as PNGs).
   - Textual summaries highlighting business insights based on the analysis.


