import requests
import io
import zipfile
import os
from tqdm import tqdm
import logging

logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("COVerAge DL")


def download_url(name, url, save_path, chunk_size=1024):
    r = requests.get(url, stream=True)
    pbar = tqdm(
        desc=f"{name} download",
        unit="B",
        total=int(r.headers["Content-Length"],),
        unit_scale=True,
    )

    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                fd.write(chunk)
                pbar.update(chunk_size)
    pbar.close()


def download_and_save_file(
    url, f_name, path="./case_data_gender_raw/", timestamp=False, overwrite=False,
):
    """
    Downloads a file and saves it to a path
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    if timestamp:
        today = datetime.datetime.today()
        f_name, extension = os.path.splitext(f_name)
        f_name = f_name + f"_{today.strftime('%m_%d')}{extension}"

    if not os.path.isfile(path + f_name) or overwrite:
        download_url(os.path.splitext(f_name)[0], url, path + f_name)
    else:
        log.warning(f"Found existing {path + f_name}, skipping download.")
    return path + f_name


if __name__ == "__main__":
    # Download coverage
    f_path = download_and_save_file("https://osf.io/9dsfk/download", "coverage.zip")

    # Unzip
    with zipfile.ZipFile(f_path, "r") as zip_ref:
        zip_ref.extractall("./case_data_gender_raw/")
