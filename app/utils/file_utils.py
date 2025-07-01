import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    return Path(filename).suffix.lower()

def is_valid_file_extension(extension: str, allowed_extensions: list) -> bool:
    """Check if a file extension is in the allowed list."""
    return extension.lower() in [ext.lower() for ext in allowed_extensions]

def save_uploaded_file(file: BinaryIO, filename: str, upload_dir: str) -> str:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        file: The file-like object to save
        filename: The original filename
        upload_dir: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file, buffer)
        
    return file_path

def create_temp_file(content: bytes, suffix: str = "") -> str:
    """
    Create a temporary file with the given content.
    
    Args:
        content: The content to write to the file
        suffix: Optional suffix for the temporary file
        
    Returns:
        Path to the created temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(content)
        return temp_file.name

def cleanup_temp_files(file_paths: list) -> None:
    """
    Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")

def get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate the hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        
    Returns:
        Hex digest of the file's hash
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    
    with open(file_path, 'rb') as f:
        file_hash = hash_func()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
            
    return file_hash.hexdigest()

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    return os.path.getsize(file_path)

def is_file_readable(file_path: str) -> bool:
    """
    Check if a file is readable.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is readable, False otherwise
    """
    return os.access(file_path, os.R_OK)
