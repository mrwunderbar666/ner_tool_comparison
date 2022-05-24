from tqdm import tqdm
import requests
from pathlib import Path

def downloader(url, destination, **kwargs):
    destination = Path(destination)
    if destination.exists():
        print('Data already downloaded, not downloading again.')
    else:
        response = requests.get(url, stream=True, **kwargs)
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit="byte") as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    b = f.write(chunk)
                    pbar.update(b)