import numpy
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re


def import_data(path):
    """
    Import data from a csv file
    """
    with open(path, "r") as file:
        text = file.read()
    # The RTF content
    rtf_content = text  # replace with your RTF content

    # Regular expression pattern for matching speakers and their dialogues
    pattern = r"\\strokec2\s(.*?)\n\\f1\\i0\s:\s(.*?)(?=\\f2\\fs24)"

    # Find all matches in the RTF content
    matches = re.findall(pattern, rtf_content, re.DOTALL)
    df = pd.DataFrame(matches, columns=["Speaker", "Dialogue"])
    return df
    # Remove the \n character from the end of each dialogue
