from pathlib import Path
from urllib.request import urlretrieve

def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "tiny_shakespeare.txt"
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if output_path.exists():
        print(f"Dataset already exists: {output_path}")
        return

    print("Downloading Tiny Shakespeare...")
    urlretrieve(url, output_path)
    print(f"Saved dataset to: {output_path}")
    
if __name__ == "__main__":
    main()
