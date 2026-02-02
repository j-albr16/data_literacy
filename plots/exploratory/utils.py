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
    try:
        return pd.to_datetime(date_series, infer_datetime_format=True)
    except:
        pass

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",     
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S%z",
    ]
   
    result = pd.to_datetime(date_series, errors='coerce')
    for fmt in formats:
        mask = result.isna()
        if not mask.any():
            break  
       
        result[mask] = pd.to_datetime(
            date_series[mask], 
            format=fmt, 
            errors='coerce'
        )
       
    return result

def add_datetime(df):
    # filter articles without date
    df = df[~df['date'].isna()]

    # add datetime by parsing dates
    df['datetime'] = parse_mixed_dates(df["date"])

    # filter articles that could not be parsed
    df = df[~df['datetime'].isna()]

    # ensure correct pd type
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # add min date and days passed
    min_date = df['datetime'].min()
    df['days_since_min'] = (df['datetime'] - min_date).dt.days

    return df
