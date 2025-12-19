import calendar
import pytz
from datetime import datetime, date, time, timedelta
from typing import Tuple, Optional

class DateUtils:
    """
    Utility class for handling option expiration dates and contract periods.
    """
    
    MILLISECONDS_IN_YEAR = 365 * 24 * 60 * 60 * 1000
    ICT_TZ = pytz.timezone('Asia/Bangkok')
    UTC_TZ = pytz.utc

    @staticmethod
    def parse_symbol_date(symbol: str) -> date:
        """Parses the date part from a Binance symbol (e.g., 'BTC-251206-C')."""
        try:
            date_str = symbol.split('-')[1]
            year = 2000 + int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return date(year, month, day)
        except IndexError:
            raise ValueError(f"Invalid symbol format: {symbol}")

    @staticmethod
    def get_expiration_type(symbol: str) -> str:
        """Determines if the option is Daily, Weekly, Monthly, or Quarterly."""
        try:
            exp_date = DateUtils.parse_symbol_date(symbol)
            is_friday = (exp_date.weekday() == 4)
            last_day_of_month = calendar.monthrange(exp_date.year, exp_date.month)[1]
            last_date_of_month = date(exp_date.year, exp_date.month, last_day_of_month)
            
            # Find the last Friday of the month
            offset = (last_date_of_month.weekday() - 4 + 7) % 7
            last_friday_day = last_date_of_month.day - offset
            
            is_last_friday_of_month = (exp_date.day == last_friday_day)
            is_quarterly_month = (exp_date.month in [3, 6, 9, 12])

            if is_last_friday_of_month and is_quarterly_month: return "Quarterly"
            elif is_last_friday_of_month: return "Monthly"
            elif is_friday: return "Weekly"
            else: return "Daily"
        except Exception: 
            return "Unknown"

    @staticmethod
    def calculate_contract_period(symbol: str, exp_type: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Calculates the start and end timestamps (ms) for a given contract.
        Returns: (start_ms, end_ms)
        """
        exp_date = DateUtils.parse_symbol_date(symbol)
        start_date = None
        
        # Expiry is at 15:00 ICT on the expiration date
        naive_end_dt = datetime.combine(exp_date, time(15, 0))
        aware_end_dt_ict = DateUtils.ICT_TZ.localize(naive_end_dt)
        
        if exp_type == "Daily": 
            start_date = exp_date - timedelta(days=1)
        elif exp_type == "Weekly": 
            start_date = exp_date - timedelta(days=7)
        elif exp_type == "Monthly":
            first = exp_date.replace(day=1)
            prev_last = first - timedelta(days=1)
            offset = (prev_last.weekday() - 4 + 7) % 7
            start_date = prev_last - timedelta(days=offset)
        elif exp_type == "Quarterly":
            m = exp_date.month
            start_m = 1 if m<=3 else 4 if m<=6 else 7 if m<=9 else 10
            first = exp_date.replace(month=start_m, day=1)
            prev_last = first - timedelta(days=1)
            offset = (prev_last.weekday() - 4 + 7) % 7
            start_date = prev_last - timedelta(days=offset)

        if start_date:
            naive_start = datetime.combine(start_date, time(15, 0))
            aware_start = DateUtils.ICT_TZ.localize(naive_start)
            
            start_ms = int(aware_start.astimezone(DateUtils.UTC_TZ).timestamp() * 1000)
            end_ms = int(aware_end_dt_ict.astimezone(DateUtils.UTC_TZ).timestamp() * 1000)
            return start_ms, end_ms
            
        return None, None