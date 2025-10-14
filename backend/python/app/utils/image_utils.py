import re
import requests
from pathlib import Path
from urllib.parse import urlparse
import base64

def get_image_bytes(image_url, markdown_file_path=None):
    """
    Get bytes for an image from URL or local file path
    
    Args:
        image_url: URL or file path to the image
        markdown_file_path: Path to the markdown file (for resolving relative paths)
    
    Returns:
        bytes: Image data as bytes
    """
    try:
        # Check if it's a web URL
        if image_url.startswith(('http://', 'https://')):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return response.content
        
        # Check if it's a data URI (base64 encoded)
        elif image_url.startswith('data:image'):
            # Format: data:image/png;base64,iVBORw0KG...
            header, encoded = image_url.split(',', 1)
            return base64.b64decode(encoded)
        
        # Otherwise, treat as local file path
        else:
            # Handle relative paths
            if markdown_file_path and not Path(image_url).is_absolute():
                markdown_dir = Path(markdown_file_path).parent
                image_path = markdown_dir / image_url
            else:
                image_path = Path(image_url)
            
            with open(image_path, 'rb') as f:
                return f.read()
    
    except Exception as e:
        print(f"Error loading image {image_url}: {e}")
        return None


def extract_images_with_bytes(markdown_text, markdown_file_path=None):
    """
    Extract all images from markdown and get their bytes
    """
    images = []
    
    # Extract Markdown-style images: ![alt](url)
    md_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    md_matches = re.findall(md_pattern, markdown_text)
    
    for alt, url in md_matches:
        img_bytes = get_image_bytes(url, markdown_file_path)
        if img_bytes:
            images.append({
                'alt': alt,
                'src': url,
                'type': 'markdown',
                'bytes': img_bytes,
                'size': len(img_bytes)
            })
    
    # Extract HTML <img> tags
    html_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
    html_matches = re.finditer(html_pattern, markdown_text)
    
    for match in html_matches:
        img_tag = match.group(0)
        src = re.search(r'src=["\']([^"\']+)["\']', img_tag).group(1)
        alt_match = re.search(r'alt=["\']([^"\']+)["\']', img_tag)
        alt = alt_match.group(1) if alt_match else ''
        
        img_bytes = get_image_bytes(src, markdown_file_path)
        if img_bytes:
            images.append({
                'alt': alt,
                'src': src,
                'type': 'html',
                'bytes': img_bytes,
                'size': len(img_bytes)
            })
    
    return images