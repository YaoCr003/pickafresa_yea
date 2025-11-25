"""
Supabase Uploader Module for Vision System

Handles uploading of captured images and JSON metadata to Supabase cloud storage.
Provides both synchronous and asynchronous upload capabilities with proper error handling.

Architecture:
    Vision Service -> SupabaseUploader -> Supabase Storage + Database
    
Tables:
    - images: Stores image metadata (id, route, timestamp)
    - json_files: Stores JSON metadata (id, route, timestamp)
    
Storage:
    - Bucket: pickafresa-captures (or custom from env)
    - Files: UUID-based naming for uniqueness
    
Usage:
    from pickafresa_vision.vision_tools.supabase_uploader import SupabaseUploader
    
    uploader = SupabaseUploader()
    
    # Synchronous upload
    result = uploader.upload_capture(
        image_path="/path/to/image.jpg",
        json_path="/path/to/data.json"
    )
    
    # Asynchronous upload (non-blocking)
    uploader.upload_capture_async(
        image_path="/path/to/image.jpg",
        json_path="/path/to/data.json",
        callback=lambda success, msg: print(f"Upload: {msg}")
    )

@aldrick-t, 2025
for Team YEA
"""

import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Repository root (pickafresa_vision/vision_tools/supabase_uploader.py -> parents[2] = repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from repo root
from dotenv import load_dotenv
env_path = REPO_ROOT / ".env"
load_dotenv(env_path)

# Supabase client (optional dependency)
SUPABASE_AVAILABLE = False
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    create_client = None
    Client = None


logger = logging.getLogger(__name__)


class SupabaseUploaderError(Exception):
    """Base exception for Supabase uploader errors."""
    pass


class SupabaseUploader:
    """
    Handles uploading of vision capture data to Supabase.
    
    Uploads images to Supabase Storage and metadata to database tables.
    Supports both synchronous and asynchronous uploads.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Supabase uploader.
        
        Args:
            supabase_url: Supabase project URL (defaults to env variable)
            supabase_key: Supabase API key (defaults to env variable)
            bucket_name: Storage bucket name (defaults to env variable)
            enabled: Enable/disable uploads
        """
        self.enabled = enabled
        self.client = None
        
        logger.debug(f"SupabaseUploader.__init__ called with enabled={enabled}")
        
        if not self.enabled:
            logger.info("Supabase uploader disabled")
            return
        
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase library not available. Install: pip install supabase")
            self.enabled = False
            return
        
        logger.debug(f"SUPABASE_AVAILABLE={SUPABASE_AVAILABLE}")
        
        # Load credentials from environment or parameters
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET", "pickafresa-captures")
        
        logger.debug(f"Credentials loaded: URL={self.supabase_url}, KEY={'***' if self.supabase_key else None}, BUCKET={self.bucket_name}")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning(f"Supabase credentials not found in environment variables (URL={bool(self.supabase_url)}, KEY={bool(self.supabase_key)})")
            self.enabled = False
            return
        
        # Initialize client
        try:
            logger.debug("Attempting to create Supabase client...")
            self.client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"✓ Supabase uploader initialized (bucket: {self.bucket_name})")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if uploader is enabled and ready."""
        return self.enabled and self.client is not None
    
    def upload_image(self, image_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Upload image to Supabase storage.
        
        Args:
            image_path: Path to image file
        
        Returns:
            (success, message, storage_path): Upload result
        """
        if not self.is_enabled():
            return False, "Supabase uploader not enabled", None
        
        if not image_path.exists():
            return False, f"Image file not found: {image_path}", None
        
        try:
            # Get file extension from actual file
            file_ext = image_path.suffix  # e.g., ".png" or ".jpg"
            content_type = "image/png" if file_ext == ".png" else "image/jpeg"
            
            # Generate unique filename with UUID and correct extension
            unique_filename = f"capturas/{uuid.uuid4()}{file_ext}"
            
            logger.info(f"Uploading image: {image_path.name} -> {unique_filename}")
            
            # Upload to storage
            with open(image_path, "rb") as f:
                result = self.client.storage.from_(self.bucket_name).upload(
                    file=f,
                    path=unique_filename,
                    file_options={"content-type": content_type}
                )
            
            logger.debug(f"Upload result: {result}")
            
            # Insert metadata into database
            data = {
                "route": unique_filename,
                "timestamp": datetime.now().isoformat()
            }
            
            db_result = self.client.table("images").insert(data).execute()
            
            logger.info(f"✓ Image uploaded: {unique_filename}")
            return True, "Image uploaded successfully", unique_filename
        
        except Exception as e:
            error_msg = f"Image upload failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def upload_json(self, json_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Upload JSON file to Supabase storage.
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            (success, message, storage_path): Upload result
        """
        if not self.is_enabled():
            return False, "Supabase uploader not enabled", None
        
        if not json_path.exists():
            return False, f"JSON file not found: {json_path}", None
        
        try:
            # Generate unique filename with UUID
            unique_filename = f"json/{uuid.uuid4()}.json"
            
            logger.debug(f"Uploading JSON: {json_path.name} -> {unique_filename}")
            
            # Read and upload JSON content
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result = self.client.storage.from_(self.bucket_name).upload(
                file=content.encode("utf-8"),
                path=unique_filename,
                file_options={"content-type": "application/json"}
            )
            
            logger.debug(f"Upload result: {result}")
            
            # Insert metadata into database
            data = {
                "route": unique_filename,
                "timestamp": datetime.now().isoformat()
            }
            
            db_result = self.client.table("json_files").insert(data).execute()
            
            logger.info(f"✓ JSON uploaded: {unique_filename}")
            return True, "JSON uploaded successfully", unique_filename
        
        except Exception as e:
            error_msg = f"JSON upload failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def upload_capture(
        self,
        image_path: Path,
        json_path: Path
    ) -> Dict[str, Any]:
        """
        Upload both image and JSON for a capture session (synchronous).
        
        Args:
            image_path: Path to image file
            json_path: Path to JSON metadata file
        
        Returns:
            Dictionary with upload results
        """
        results = {
            "success": False,
            "image_success": False,
            "json_success": False,
            "image_path": None,
            "json_path": None,
            "error": None
        }
        
        if not self.is_enabled():
            results["error"] = "Supabase uploader not enabled"
            return results
        
        # Upload image
        img_success, img_msg, img_path = self.upload_image(image_path)
        results["image_success"] = img_success
        results["image_path"] = img_path
        
        # Upload JSON
        json_success, json_msg, json_path_result = self.upload_json(json_path)
        results["json_success"] = json_success
        results["json_path"] = json_path_result
        
        # Overall success requires both uploads
        results["success"] = img_success and json_success
        
        if not results["success"]:
            errors = []
            if not img_success:
                errors.append(f"Image: {img_msg}")
            if not json_success:
                errors.append(f"JSON: {json_msg}")
            results["error"] = "; ".join(errors)
        
        return results
    
    def upload_capture_async(
        self,
        image_path: Path,
        json_path: Path,
        callback: Optional[Callable[[bool, str, Dict[str, Any]], None]] = None
    ) -> threading.Thread:
        """
        Upload both image and JSON asynchronously (non-blocking).
        
        Args:
            image_path: Path to image file
            json_path: Path to JSON metadata file
            callback: Optional callback function(success, message, results)
        
        Returns:
            Thread object (already started)
        """
        def _upload_thread():
            results = self.upload_capture(image_path, json_path)
            
            if callback:
                try:
                    callback(
                        results["success"],
                        results.get("error", "Upload complete"),
                        results
                    )
                except Exception as e:
                    logger.error(f"Upload callback error: {e}")
        
        thread = threading.Thread(target=_upload_thread, daemon=True)
        thread.start()
        return thread


def main():
    """Test Supabase uploader with sample files."""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Test Supabase uploader")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--json", type=str, help="Path to test JSON")
    parser.add_argument("--async", action="store_true", help="Use async upload")
    
    args = parser.parse_args()
    
    uploader = SupabaseUploader()
    
    if not uploader.is_enabled():
        print("❌ Supabase uploader not enabled. Check credentials in .env")
        return 1
    
    if args.image and args.json:
        image_path = Path(args.image)
        json_path = Path(args.json)
        
        if getattr(args, "async"):
            print("Starting async upload...")
            
            def callback(success, message, results):
                if success:
                    print(f"✓ Upload complete: {message}")
                else:
                    print(f"✗ Upload failed: {message}")
            
            thread = uploader.upload_capture_async(image_path, json_path, callback)
            thread.join()  # Wait for completion
        else:
            print("Starting sync upload...")
            results = uploader.upload_capture(image_path, json_path)
            
            if results["success"]:
                print(f"✓ Upload complete")
                print(f"  Image: {results['image_path']}")
                print(f"  JSON: {results['json_path']}")
            else:
                print(f"✗ Upload failed: {results['error']}")
    else:
        print("Please provide --image and --json paths")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
