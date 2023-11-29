import numpy
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


def import_data(path):
    """
    Import data from a csv file
    """

    with open(path, "r") as file:
        text = file.read()

    lines = text.split("\n")

    speakers = []
    dialogues = []

    for line in lines:
        parts = line.split(": ")
        if len(parts) >= 2:
            speaker = parts[0].strip()
            dialogue = ": ".join(parts[1:]).strip()

            speakers.append(speaker)
            dialogues.append(dialogue)

    data = {"Speaker": speakers, "Dialogue": dialogues}
    df = pd.DataFrame(data)

    return df
