import re
import requests


def download_dickinson_text(url, save_path="dickinson_raw.txt"):
    print(f"Downloading from {url}...")
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download data. Status code: {response.status_code}"
        )
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Raw text saved to {save_path}")
    return save_path


def remove_gutenberg_headers(raw_text):
    pattern = r"(?:\*{3}\s*START OF.*?\*{3})(.*?)(?:\*{3}\s*END OF.*?\*{3})"
    match = re.search(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        main_text = match.group(1)
    else:
        main_text = raw_text
    return main_text


def skip_until_first_poem(lines):
    cleaned = []
    skipping = True

    poem_start_triggers = [
        "LIFE",
        "PART ONE: LIFE",
        "I. LIFE.",
        "POEMS",
    ]

    for line in lines:
        line_stripped = line.strip().upper()
        if any(trigger in line_stripped for trigger in poem_start_triggers):
            skipping = False
            continue
        if not skipping:
            cleaned.append(line)
    return cleaned


def is_potential_title(line):
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[IVXLCDM]+\.$", stripped.upper().replace(" ", "")):
        return True
    letters = re.findall(r"[a-zA-Z]", stripped)
    if not letters:
        return True
    uppercase_letters = sum(1 for c in letters if c.isupper())
    ratio = uppercase_letters / len(letters)
    word_count = len(stripped.split())
    if ratio >= 0.8 and word_count <= 6:
        return True
    return False


def remove_titles(lines):
    cleaned = []
    end_token = "<END>"
    placed_end_token = False
    for line in lines:
        if is_potential_title(line):
            if not placed_end_token:
                cleaned.extend([end_token, ""])
                placed_end_token = True
            continue
        cleaned.append(line)
        placed_end_token = False
    return cleaned


def remove_extra_sections(lines):
    return lines


def basic_line_cleaning(lines):
    cleaned = []
    empty = False
    for line in lines:
        line = line.strip()
        if not line:
            if empty:
                continue
            empty = True
        else:
            empty = False
        cleaned.append(line)
    return cleaned


def main():
    url = "https://www.gutenberg.org/files/12242/12242-0.txt"
    raw_file_path = "dickinson_raw.txt"
    cleaned_file_path = "dickinson_clean.txt"

    download_dickinson_text(url, raw_file_path)
    with open(raw_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    main_text = remove_gutenberg_headers(raw_text)
    lines = main_text.split("\n")
    lines = skip_until_first_poem(lines)
    lines = remove_titles(lines)
    lines = remove_extra_sections(lines)
    lines = basic_line_cleaning(lines)
    cleaned_text = "\n".join(lines)
    with open(cleaned_file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"Cleaned text saved to {cleaned_file_path}.")
    print("Time for a more thorough manual cleaning!")


if __name__ == "__main__":
    main()
