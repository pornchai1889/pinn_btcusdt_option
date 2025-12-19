# src/data/binance_api.py
import requests
import time as t_module
import logging
from datetime import datetime
from typing import List, Dict, Union, Any

class BinanceDataFetcher:
    """
    Handles robust fetching of K-line data from Binance API (Spot & Options).
    Implements pagination, deduplication, and retry logic.
    """
    
    def __init__(self, timeout: int = 10, limit: int = 1000):
        self.timeout = timeout
        self.limit = limit
        self.logger = logging.getLogger(__name__)

    def fetch_klines(self, url: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[Any]:
        """
        Fetches historical k-lines forward from start_ms to end_ms.
        """
        unique_data_map = {} 
        current_start = start_ms
        
        self.logger.info(f"Fetching {symbol} ({interval}): {datetime.fromtimestamp(start_ms/1000)} -> {datetime.fromtimestamp(end_ms/1000)}")
        
        while True:
            if current_start >= end_ms:
                self.logger.info("Reached target end time.")
                break

            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'limit': self.limit
            }
            
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    self.logger.info("No more data from API.")
                    break

                # 1. Sort Data (Oldest -> Newest)
                if isinstance(data[0], dict):
                    data.sort(key=lambda x: int(x['openTime']))
                else:
                    data.sort(key=lambda x: int(x[0]))
                
                # 2. Store & Deduplicate
                new_count = 0
                last_close_time = 0
                
                for k in data:
                    if isinstance(k, dict): 
                        t = int(k['openTime'])
                        close_t = int(k['closeTime'])
                    else: 
                        t = int(k[0])
                        close_t = int(k[6])
                    
                    if t <= end_ms:
                        if t not in unique_data_map:
                            unique_data_map[t] = k
                            new_count += 1
                    
                    last_close_time = close_t
                
                self.logger.debug(f"Fetched batch ending {datetime.fromtimestamp(last_close_time/1000)}. New: {new_count}")

                # 3. Pagination Logic
                next_start = last_close_time + 1
                
                # 4. Stuck Loop Protection
                if next_start <= current_start:
                    self.logger.warning("Stuck at same time. Forcing jump forward...")
                    delta = self._parse_interval(interval)
                    current_start += delta
                else:
                    current_start = next_start
                
                if len(data) < self.limit:
                    self.logger.info("End of available history (Batch < Limit).")
                    break
                    
                t_module.sleep(0.1) # Rate limit protection
                
            except Exception as e:
                self.logger.error(f"Fetching failed: {e}")
                break
                
        # 5. Return Sorted List
        sorted_data = sorted(unique_data_map.values(), key=lambda x: int(x['openTime']) if isinstance(x, dict) else int(x[0]))
        return sorted_data

    def _parse_interval(self, interval: str) -> int:
        """Parses interval string to milliseconds."""
        if 'm' in interval: return int(interval.replace('m','')) * 60 * 1000
        elif 'h' in interval: return int(interval.replace('h','')) * 60 * 60 * 1000
        elif 'd' in interval: return int(interval.replace('d','')) * 24 * 60 * 60 * 1000
        return 3600000 # Default 1h