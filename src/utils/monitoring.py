"""
Monitoring utilities for resource and process tracking.
"""

from collections import defaultdict
import time
import asyncio
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ProcessingMonitor:
    """Monitors document processing tasks."""
    
    def __init__(self):
        self.start_times = defaultdict(dict)
        self.end_times = defaultdict(dict)
        self.active_tasks = 0
        self._lock = asyncio.Lock()
    
    async def start_task(self, file_path: str):
        """Record task start time."""
        async with self._lock:
            self.start_times[file_path] = time.time()
            self.active_tasks += 1
            logger.info(
                f"Started processing {file_path}. "
                f"Active tasks: {self.active_tasks}"
            )
    
    async def end_task(self, file_path: str):
        """Record task end time."""
        async with self._lock:
            self.end_times[file_path] = time.time()
            self.active_tasks -= 1
            duration = self.end_times[file_path] - self.start_times[file_path]
            logger.info(
                f"Finished processing {file_path} in {duration:.2f}s. "
                f"Active tasks: {self.active_tasks}"
            )
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        total_files = len(self.end_times)
        if total_files == 0:
            return {
                "total_files": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": 0.0,
                "total_duration": 0.0
            }
            
        durations = [
            self.end_times[f] - self.start_times[f] 
            for f in self.end_times.keys()
        ]
        
        return {
            "total_files": total_files,
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_duration": max(self.end_times.values()) - min(self.start_times.values())
        } 