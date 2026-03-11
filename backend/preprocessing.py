import pandas as pd
import os
import re


STOPWORDS = {
    "the","i","am","having","with","my","issue","product",
    "a","an","is","are","this","that","it","to","for",
    "please","help","regarding","request","purchased",
    "recently","make","making","ve","m","hi","hello"
}


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9 ]", " ", text)

    words = text.split()

    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)


def load_and_clean_data():

    print("Reading dataset...\n")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "support_tickets.csv")

    df = pd.read_csv(file_path)

    df = df.fillna("")

    # shorten description to remove template sentences
    desc = df["Ticket Description"].apply(
        lambda x: " ".join(str(x).split()[:10])
    )

    df["text"] = (
        df["Ticket Subject"] + " " +
        df["Ticket Type"] + " " +
        df["Product Purchased"] + " " +
        desc
    )

    df["text"] = df["text"].apply(clean_text)

    df = df.drop_duplicates(subset=["text"])

    return df