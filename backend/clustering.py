import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from collections import Counter

from backend.preprocessing import load_and_clean_data


IGNORE_WORDS = {
    "assist","inquiry","problem","issue","request","help",
    "customer","support","team","service","ticket",
    "product","technical","in","unable",

    # brand names
    "lg","samsung","amazon","canon","nintendo",
    "philips","roomba","adobe","playstation"
}


def generate_issue_name(texts):
    """
    Generate issue title from most common keywords
    """

    words = " ".join(texts).split()

    words = [w for w in words if w not in IGNORE_WORDS]

    common = Counter(words).most_common(2)

    return " ".join([w[0] for w in common])


def detect_issue_clusters():

    print("Loading dataset...\n")

    df = load_and_clean_data()

    texts = df["text"].tolist()

    print("Creating TF-IDF vectors...\n")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1,2),
        min_df=5
    )

    X = vectorizer.fit_transform(texts)

    print("Reducing dimensionality (SVD)...\n")

    svd = TruncatedSVD(
        n_components=100,
        random_state=42
    )

    X_reduced = svd.fit_transform(X)

    print("Clustering tickets...\n")

    k = 12

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_reduced)

    df["cluster"] = clusters

    issues = []

    for cluster_id in sorted(df["cluster"].unique()):

        cluster_df = df[df["cluster"] == cluster_id]

        mentions = len(cluster_df)

        if mentions < 20:
            continue

        issue_name = generate_issue_name(cluster_df["text"])

        # Improvement 1: remove duplicate example tickets
        examples = cluster_df["Ticket Subject"].drop_duplicates().head(3).tolist()

        issues.append({
            "issue": issue_name,
            "mentions": mentions,
            "examples": examples
        })

    # Improvement 2: merge duplicate issue titles
    merged = {}

    for issue in issues:

        name = issue["issue"]

        if name not in merged:
            merged[name] = issue
        else:
            merged[name]["mentions"] += issue["mentions"]
            merged[name]["examples"].extend(issue["examples"])

    issues = sorted(
        merged.values(),
        key=lambda x: x["mentions"],
        reverse=True
    )

    print("\nDetected Issue Clusters\n")

    for issue in issues:

        print(f"Issue: {issue['issue']}")
        print(f"Mentions: {issue['mentions']}")

        print("Example Tickets:")

        for e in issue["examples"][:3]:
            print("-", e)

        print("\n-------------------------")

    return df


if __name__ == "__main__":
    detect_issue_clusters()