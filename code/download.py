import os
import s3fs
from tqdm import tqdm

class Downloader:
    def __init__(self, anon=True, **kwargs):
        self.fs = s3fs.S3FileSystem(anon=anon, **kwargs)

    def read(self, url):
        with self.fs.open(url, "rb") as f:
            return f.read()

    def download(self, url, filename):
        with self.fs.open(url, "rb") as file_in:
            with open(filename, "wb") as file_out:
                file_out.write(file_in.read())

    def download_list(self, url_list, folder, show_progress=True, skip=True):
        if skip and os.path.isdir(folder):
            names_in_folder = [f.name for f in os.scandir(folder)]
            url_list = [url for url in url_list 
                if os.path.basename(url) not in names_in_folder]
        
        os.makedirs(folder, exist_ok=True)
        if show_progress:
            progress = tqdm(total=len(url_list))
        for url in url_list:
            name = os.path.basename(url)
            filename = os.path.join(folder, name)
            self.download(url, filename)
            if show_progress:
                progress.update()
        if show_progress:
            progress.close()
        return