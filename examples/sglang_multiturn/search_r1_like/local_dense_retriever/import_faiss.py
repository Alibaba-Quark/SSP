def import_faiss():
    from pathlib import Path

    import pip
    import requests
    from tqdm import tqdm

    url = "https://githubfast.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    filename = Path(url).name
    headers = {"User-Agent": "Wget/1.21.3"}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    with open(filename, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                file.write(chunk)

    pip.main(["install", filename])
    print("faiss installed successfully.")


if __name__ == "__main__":
    try:
        import faiss
    except Exception:
        import_faiss()
