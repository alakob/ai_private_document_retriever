"""
File handling utilities.
"""

import os
import logging
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional
from ..utils.errors import FileLoadError

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations with validation and error checking."""
    
    def __init__(self, upload_dir: str = "documents"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        self.supported_extensions = {
            '.pdf', '.txt', '.doc', '.docx', '.ppt', '.pptx'
        }
    
    def validate_file(self, file_path: Path) -> None:
        """Validate file before processing."""
        if not file_path.exists():
            raise FileLoadError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() not in self.supported_extensions:
            raise FileLoadError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )
            
        if file_path.stat().st_size == 0:
            raise FileLoadError(f"File is empty: {file_path}")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def safe_move_file(
        self,
        source_path: Path,
        dest_dir: Optional[Path] = None
    ) -> Path:
        """Safely move file to destination directory."""
        dest_dir = dest_dir or self.upload_dir
        dest_path = dest_dir / source_path.name
        
        # Handle duplicate filenames
        counter = 1
        while dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            new_name = f"{stem}_{counter}{suffix}"
            dest_path = dest_dir / new_name
            counter += 1
        
        shutil.move(str(source_path), str(dest_path))
        return dest_path
    
    def cleanup_old_files(
        self,
        max_age_days: int = 30,
        exclude_patterns: Optional[List[str]] = None
    ) -> None:
        """Clean up old files from upload directory."""
        exclude_patterns = exclude_patterns or []
        current_time = time.time()
        
        for file_path in self.upload_dir.glob('*.*'):
            # Skip excluded files
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
                
            # Check file age
            file_age_days = (current_time - file_path.stat().st_mtime) / (24 * 3600)
            if file_age_days > max_age_days:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}") 