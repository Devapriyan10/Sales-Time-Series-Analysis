import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load dataset
def load_data(file_path):
    """Load sales data from a CSV file."""
    try:
        date_format = "%d-%m-%Y %H:%M"
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp', date_parser=lambda x: pd.to_datetime(x, format=date_format))
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        raise
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the data.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

# Resample data
def resample_data(df, freq):
    """Resample data to a specified frequency."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    return df.resample(freq).sum()

# Decompose time series
def decompose_series(series, model='additive', period=None):
    """Decompose a time series into its components."""
    if period is None:
        period = 365 if series.index.freq == 'D' else 12  # Default to yearly for daily, monthly for others
    return seasonal_decompose(series, model=model, period=period)

# Identify low and high sales periods
def identify_low_high_sales(series, threshold=0.25):
    """Identify periods of low and high sales."""
    sorted_series = series.sort_values()
    low_sales = sorted_series[:int(len(sorted_series) * threshold)]
    high_sales = sorted_series[-int(len(sorted_series) * threshold):]
    return low_sales, high_sales

# Print decomposition results, identify trends, and generate plots
def print_decomposition(result, title):
    """Generate and save plots and textual summaries based on the decomposition results."""
    summary = f"\n{title}:\n"
    
    # Identify low and high sales periods
    low_sales, high_sales = identify_low_high_sales(result.observed)
    summary += f"Periods of Low Sales:\n{low_sales}\n\n"
    summary += f"Periods of High Sales:\n{high_sales}\n\n"

    # Generate and save plots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    ax1.plot(result.observed.index, result.observed, label='Observed', color='blue')
    ax1.set_ylabel('Observed')
    ax1.set_title(f'{title} - Observed')
    ax1.legend()

    ax2.plot(result.trend.index, result.trend, label='Trend', color='orange')
    ax2.set_ylabel('Trend')
    ax2.set_title(f'{title} - Trend')
    ax2.legend()

    ax3.plot(result.seasonal.index, result.seasonal, label='Seasonal', color='green')
    ax3.set_ylabel('Seasonal')
    ax3.set_title(f'{title} - Seasonal')
    ax3.legend()

    ax4.plot(result.resid.index, result.resid, label='Residual', color='red')
    ax4.set_ylabel('Residual')
    ax4.set_title(f'{title} - Residual')
    ax4.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

    # Business improvement insights based on trend and seasonality
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    
    if 'Hourly Analysis' in title:
        forecasted_peak_hours = trend.idxmax()
        forecasted_low_hours = trend.idxmin()
        
        summary += f"Insights for Improving Business:\n"
        summary += f"- **Peak Hours Forecasted:** Based on current trends, the peak sales hours are predicted around {forecasted_peak_hours.strftime('%H:%M')}. Ensure optimal staffing and stock levels during these hours to maximize revenue.\n"
        summary += f"- **Low Sales Hours Forecasted:** The hours with the lowest sales are predicted around {forecasted_low_hours.strftime('%H:%M')}. Implement targeted promotions or discounts during these periods to boost sales.\n"
        summary += f"- **Trend Analysis:** Adjust operational hours and marketing strategies based on the hourly trend analysis. For instance, if sales consistently dip at certain hours, consider revising store hours or running promotions to attract customers during off-peak times.\n"

    elif 'Daily Analysis' in title:
        forecasted_high_sales_day = trend.idxmax()
        forecasted_low_sales_day = trend.idxmin()
        
        summary += f"Insights for Improving Business:\n"
        summary += f"- **Peak Sales Days Forecasted:** The day of the week with the highest sales is predicted to be {forecasted_high_sales_day.strftime('%A')}. Plan special offers or events on this day to further boost sales.\n"
        summary += f"- **Low Sales Days Forecasted:** The day of the week with the lowest sales is predicted to be {forecasted_low_sales_day.strftime('%A')}. Consider running targeted promotions or discounts to improve sales on this day.\n"
        summary += f"- **Trend Analysis:** Analyze daily sales trends to optimize inventory and staffing levels. Ensure that high-traffic days are well-stocked and adequately staffed. Tailor marketing efforts to attract customers on slower days.\n"

    elif 'Weekly Analysis' in title:
        forecasted_high_sales_week = trend.idxmax()
        forecasted_low_sales_week = trend.idxmin()
        
        summary += f"Insights for Improving Business:\n"
        summary += f"- **Peak Weeks Forecasted:** The week with the highest sales is predicted around {forecasted_high_sales_week.strftime('%Y-%m-%d')}. Plan inventory and staffing levels accordingly to handle increased demand.\n"
        summary += f"- **Low Sales Weeks Forecasted:** The week with the lowest sales is predicted around {forecasted_low_sales_week.strftime('%Y-%m-%d')}. Use this information to implement promotions or discounts to boost sales during these periods.\n"
        summary += f"- **Trend Analysis:** Adjust marketing strategies based on weekly trends. Implement weekly promotions to address sales fluctuations and prepare for expected high-sales weeks.\n"

    elif 'Monthly Analysis' in title:
        forecasted_high_sales_month = trend.idxmax()
        forecasted_low_sales_month = trend.idxmin()
        
        summary += f"Insights for Improving Business:\n"
        summary += f"- **Peak Sales Months Forecasted:** The month with the highest sales is predicted to be {forecasted_high_sales_month.strftime('%B')}. Prepare for this peak by increasing inventory and launching marketing campaigns.\n"
        summary += f"- **Low Sales Months Forecasted:** The month with the lowest sales is predicted to be {forecasted_low_sales_month.strftime('%B')}. Plan targeted promotions and adjust inventory to address potential low sales.\n"
        summary += f"- **Trend Analysis:** Utilize monthly trends to set realistic sales targets and adjust marketing strategies. Prepare for seasonal variations in sales and optimize inventory management based on predicted trends.\n"

    elif 'Seasonal Analysis' in title:
        forecasted_high_sales_season = seasonal.idxmax()
        forecasted_low_sales_season = seasonal.idxmin()
        
        summary += f"Insights for Improving Business:\n"
        summary += f"- **Peak Seasons Forecasted:** The season with the highest sales is predicted to be {forecasted_high_sales_season.strftime('%B')}. Plan for major events and holidays during this period by increasing inventory and running targeted marketing campaigns.\n"
        summary += f"- **Low Seasons Forecasted:** The season with the lowest sales is predicted to be {forecasted_low_sales_season.strftime('%B')}. Implement strategies to boost sales during these periods, such as discounts or special promotions.\n"
        summary += f"- **Trend Analysis:** Use seasonal patterns to optimize year-round business performance. Develop long-term strategies based on predicted high and low seasons to maximize sales and manage inventory effectively.\n"

    return summary

# Perform analysis and generate text reports
def analyze_data(file_path):
    """Analyze sales data and generate reports and plots."""
    df = load_data(file_path)
    summaries = []

    # Hourly Analysis
    hourly_data = resample_data(df, 'H')
    if len(hourly_data) >= 48:
        hourly_decompose = decompose_series(hourly_data['sales'], period=24)
        summaries.append(print_decomposition(hourly_decompose, 'Hourly Analysis'))
    else:
        summaries.append("Not enough data for hourly analysis")

    # Daily Analysis
    daily_data = resample_data(df, 'D')
    if len(daily_data) >= 14:
        daily_decompose = decompose_series(daily_data['sales'], period=7)
        summaries.append(print_decomposition(daily_decompose, 'Daily Analysis'))
    else:
        summaries.append("Not enough data for daily analysis")

    # Weekly Analysis
    weekly_data = resample_data(df, 'W')
    if len(weekly_data) >= 8:
        weekly_decompose = decompose_series(weekly_data['sales'], period=4)
        summaries.append(print_decomposition(weekly_decompose, 'Weekly Analysis'))
    else:
        summaries.append("Not enough data for weekly analysis")

    # Monthly Analysis
    monthly_data = resample_data(df, 'M')
    if len(monthly_data) >= 24:
        monthly_decompose = decompose_series(monthly_data['sales'], period=12)
        summaries.append(print_decomposition(monthly_decompose, 'Monthly Analysis'))
    else:
        summaries.append("Not enough data for monthly analysis")

    # Seasonal Analysis (assuming seasonality is yearly and we have monthly data)
    if len(monthly_data) >= 24:
        seasonal_decompose_result = decompose_series(monthly_data['sales'], period=12)
        summaries.append(print_decomposition(seasonal_decompose_result, 'Seasonal Analysis'))
    else:
        summaries.append("Not enough data for seasonal analysis")

    # Print all summaries
    for summary in summaries:
        print(summary)

if __name__ == "__main__":
    file_path = 'D:/sales-ai-analysis/OneDrive_2024-10-15/Sales Analytics-11(correct)/sufficient_sales_data_two_years.csv'  # Ensure this path is correct
    analyze_data(file_path)
