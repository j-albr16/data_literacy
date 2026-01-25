import pandas as pd


def add_bins(bins):
    def bin_value(value):
        for i, bin in enumerate(bins):
            if value <=bin:
                return i

        return len(bins)
    return bin_value



def parse_mixed_dates(date_series):
    """
    Parse a pandas Series with mixed date formats.
    Returns a datetime Series.
    """
    # Try pandas first - it's very good at inferring formats
    try:
        return pd.to_datetime(date_series, infer_datetime_format=True)
    except:
        pass
   
    # If that fails, try format by format
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",      # ISO 8601 with timezone (no colon)
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S%z",
    ]
   
    # Use pd.to_datetime with errors='coerce' to handle each row
    result = pd.to_datetime(date_series, errors='coerce')
    for fmt in formats:
        # Only try to parse rows that haven't been parsed yet
        mask = result.isna()
        if not mask.any():
            break  # All parsed, we're done
       
        # Try this format on unparsed rows
        result[mask] = pd.to_datetime(
            date_series[mask], 
            format=fmt, 
            errors='coerce'
        )
       
    return result

def add_datetime(df):
    # check how many newspapers do not have a date
    print(f'{df['date'].isna().sum()} articles do not have a date. They are filtered out')
    df = df[~df['date'].isna()]


    # parsing dates to datetimes
    df['datetime'] = parse_mixed_dates(df["date"])
    print(f"Failed to parse: {df['datetime'].isna().sum()} rows")
    print(df[df['datetime'].isna()]['date'].head())

    # Filter out rows where datetime parsing failed
    df = df[~df['datetime'].isna()]

    # Convert to timezone-aware datetime
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Find minimum date
    min_date = df['datetime'].min()
    print(f"Minimum date: {min_date}")

    # Calculate days since minimum date
    df['days_since_min'] = (df['datetime'] - min_date).dt.days

    print(f"Date range: {df['days_since_min'].min()} to {df['days_since_min'].max()} days")


    return df
