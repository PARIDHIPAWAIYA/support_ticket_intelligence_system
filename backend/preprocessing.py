import pandas as pd
import re

def load_and_clean_data():

    print("Loading dataset...")

    df = pd.read_csv("data/support_tickets.csv")

    columns = [
        "Ticket Subject",
        "Ticket Description",
        "Ticket Type",
        "Product Purchased",
        "Date of Purchase",
        "Ticket Priority",
        "Ticket Channel"
    ]

    df = df[columns]

    df = df.fillna("")

    df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]

    df["text"] = df["text"].str.lower()

    df["text"] = df["text"].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x))

    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], dayfirst=True)

    print("Preprocessing completed!\n")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())