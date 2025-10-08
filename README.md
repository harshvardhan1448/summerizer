# Summarizer

A powerful and easy-to-use text summarization tool that helps you condense lengthy articles, documents, or any text into concise and meaningful summaries.

## Features

- **Automatic Text Summarization:** Quickly generate summaries from long-form content.
- **User-Friendly Interface:** Simple input and output for seamless summarization.
- **Customizable Summary Length:** Adjust the size of the summary to fit your needs.
- **Multi-language Support:** Summarizes text in various languages (if applicable).
- **Open Source:** Free to use, modify, and contribute.

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/harshvardhan1448/summerizer.git
cd summerizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Command-Line Interface

```bash
python summarize.py input.txt
```

Replace `input.txt` with the path to your text file. The summary will be printed to the console or saved to an output file if specified.

#### As a Python Module

```python
from summarizer import summarize

text = "Your long text here."
summary = summarize(text, ratio=0.2)  # Adjust ratio as needed
print(summary)
```

### Configuration

- `ratio`: Float between 0 and 1 indicating the proportion of the original text to include in the summary.
- `language`: (Optional) Specify the language of the text for better results.

## Examples

#### Summarizing via CLI

```bash
python summarize.py --input article.txt --output summary.txt --ratio 0.2
```

#### Summarizing in Python

```python
summary = summarize("This is a long article...", ratio=0.15)
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgments

- Inspired by popular NLP and summarization libraries.
- Thank you to all contributors and open-source projects that made this possible.

---
**Note:** If you encounter any issues or have feature requests, please open an issue on the [GitHub repository](https://github.com/harshvardhan1448/summerizer/issues).
