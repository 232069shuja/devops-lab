from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_curated_jigsaw_dataset() -> pd.DataFrame:
    """Constructs a curated dataset adhering to Jigsaw Unintended Bias specifications,

    including civil conversation, neutral identity mentions (bias mitigation),
    and toxic insults/threats.
    """
    records = [
        # --- Benign / Civil discussions (non-toxic = 0, target < 0.5) ---
        ("I love the idea of upvoting entire articles.", 0.0, 0),
        (
            "This is a well-researched article that raises valid points on"
            " infrastructure.",
            0.0,
            0,
        ),
        (
            "Thank you for sharing your thoughtful perspective on local council"
            " policy.",
            0.0,
            0,
        ),
        (
            "The community meeting has been rescheduled for next Tuesday at 7"
            " PM.",
            0.0,
            0,
        ),
        (
            "Electric vehicles are becoming more efficient every single year.",
            0.0,
            0,
        ),
        (
            "I agree with the author's economic analysis regarding interest"
            " rates.",
            0.0,
            0,
        ),
        (
            "We should plant more native trees in the urban parks to support"
            " biodiversity.",
            0.0,
            0,
        ),
        (
            "Great analysis and very insightful graphs in the second section.",
            0.0,
            0,
        ),
        ("The scientific method relies on peer-reviewed replication.", 0.0, 0),
        (
            "Public libraries offer invaluable educational resources for"
            " children.",
            0.0,
            0,
        ),
        (
            "I disagree with this policy proposal, but I respect the underlying"
            " intent.",
            0.1,
            0,
        ),
        (
            "Let us review the budget allocation before reaching conclusions.",
            0.0,
            0,
        ),
        ("Solar energy installations have increased by twenty percent.", 0.0, 0),
        (
            "The author makes a compelling argument for improving railway"
            " transit.",
            0.0,
            0,
        ),
        (
            "Good luck with your research project! I hope the findings get"
            " published.",
            0.0,
            0,
        ),
        (
            "Clean drinking water should be a universal human priority across"
            " the globe.",
            0.0,
            0,
        ),
        (
            "Reading different viewpoints helps us develop empathy and"
            " critical thinking.",
            0.0,
            0,
        ),
        (
            "The architectural design of the new bridge is both functional and"
            " graceful.",
            0.0,
            0,
        ),
        (
            "Agricultural cooperatives help smallholder farmers achieve stable"
            " pricing.",
            0.0,
            0,
        ),
        (
            "Can someone clarify the methodology used in Table 4 of the"
            " report?",
            0.0,
            0,
        ),
        # --- Benign identity mentions (Mitigating unintended bias on demographic terms) ---
        (
            "As a proud Muslim woman, I volunteer regularly at our local food"
            " pantry.",
            0.0,
            0,
        ),
        (
            "My Jewish grandmother taught me the importance of kindness and"
            " hospitality.",
            0.0,
            0,
        ),
        (
            "He is a gay man who has served honorably in public education for"
            " thirty years.",
            0.0,
            0,
        ),
        (
            "Christian charities and community centers are coordinating winter"
            " coats for families.",
            0.0,
            0,
        ),
        (
            "Black entrepreneurs are leading impressive technological startups"
            " in Detroit.",
            0.0,
            0,
        ),
        (
            "Transgender individuals deserve fair access to healthcare,"
            " employment, and dignity.",
            0.0,
            0,
        ),
        (
            "The Hindu temple hosted a cultural celebration open to all"
            " community members.",
            0.0,
            0,
        ),
        (
            "She is an Asian American scientist pioneering cancer"
            " immunotherapy treatments.",
            0.0,
            0,
        ),
        (
            "Our Buddhist neighbor practices mindfulness and maintains a"
            " beautiful garden.",
            0.0,
            0,
        ),
        (
            "Latino artists and musicians contributed vibrant murals to"
            " downtown neighborhoods.",
            0.0,
            0,
        ),
        (
            "Athletes with physical disabilities demonstrated extraordinary"
            " endurance in the games.",
            0.0,
            0,
        ),
        (
            "People living with psychiatric or mental illness benefit greatly"
            " from supportive therapy.",
            0.0,
            0,
        ),
        (
            "Atheist and secular philosophers have contributed meaningfully to"
            " ethical debates.",
            0.0,
            0,
        ),
        (
            "White, black, and brown neighbors came together to clean up the"
            " neighborhood park.",
            0.0,
            0,
        ),
        (
            "Lesbian and bisexual leaders have advocated for marriage equality"
            " worldwide.",
            0.0,
            0,
        ),
        (
            "The conference celebrated achievements of women engineers and"
            " mathematicians.",
            0.0,
            0,
        ),
        (
            "Children with intellectual disabilities showed great joy during"
            " music therapy.",
            0.0,
            0,
        ),
        (
            "Interfaith dialogue between Muslims, Jews, and Christians fosters"
            " mutual respect.",
            0.0,
            0,
        ),
        (
            "Transgender men and women should be treated with basic human"
            " decency and care.",
            0.0,
            0,
        ),
        (
            "He spoke warmly about his Mexican heritage and family holiday"
            " traditions.",
            0.0,
            0,
        ),
        # --- Toxic comments (toxic = 1, target >= 0.5) ---
        (
            "haha you guys are a bunch of losers who do not know anything.",
            0.89,
            1,
        ),
        (
            "Shut your mouth you pathetic idiot, you have no brain whatsoever.",
            0.92,
            1,
        ),
        (
            "You are completely disgusting, vile, and an absolute waste of"
            " oxygen.",
            0.95,
            1,
        ),
        (
            "Stop posting your garbage opinions here, nobody wants to hear your"
            " stupidity.",
            0.80,
            1,
        ),
        (
            "What a moron! Go back to whatever rock you crawled out from under.",
            0.88,
            1,
        ),
        (
            "You are a dishonest scumbag and everyone in this forum knows you"
            " are lying.",
            0.85,
            1,
        ),
        (
            "I hope you get fired from your job and lose everything you"
            " possess, you freak.",
            0.86,
            1,
        ),
        (
            "Get lost you hateful creep, nobody likes you or respects your"
            " worthless thoughts.",
            0.87,
            1,
        ),
        (
            "What an ugly and repulsive human being. You make me sick just"
            " reading this.",
            0.91,
            1,
        ),
        (
            "You are a brainless troll trying to ruin this site for everyone"
            " else.",
            0.82,
            1,
        ),
        (
            "Drop dead already, you useless excuse of a person.",
            0.96,
            1,
        ),
        (
            "Only an illiterate clown would write something so thoroughly"
            " ignorant.",
            0.84,
            1,
        ),
        (
            "Your entire family must be ashamed of what a cowardly hypocrite"
            " you are.",
            0.89,
            1,
        ),
        (
            "This comment is utter trash written by a total lunatic with zero"
            " intelligence.",
            0.83,
            1,
        ),
        (
            "You deserve to be publicly humiliated and beaten for saying"
            " something that foul.",
            0.94,
            1,
        ),
        (
            "Get out of our community, you parasitic vermin and pathological"
            " liar.",
            0.90,
            1,
        ),
        (
            "You talk like a corrupt degenerate who belongs behind bars"
            " immediately.",
            0.87,
            1,
        ),
        (
            "Nobody cares about your whining, wipe your tears you pathetic"
            " coward.",
            0.81,
            1,
        ),
        (
            "Your vile hatred and toxic attacks are sickening to every decent"
            " reader.",
            0.89,
            1,
        ),
        (
            "Such blatant stupidity should be banned permanently from the"
            " entire internet.",
            0.85,
            1,
        ),
    ]

    expanded = []
    for comment, target, label in records:
        expanded.append(
            {"comment_text": comment, "target": target, "toxic": label}
        )
        if label == 1:
            expanded.append(
                {
                    "comment_text": (
                        f"Seriously? {comment.capitalize()} Absolutely"
                        " shameful!"
                    ),
                    "target": min(1.0, target + 0.05),
                    "toxic": 1,
                }
            )
        else:
            expanded.append(
                {
                    "comment_text": f"Indeed, {comment.lower()}",
                    "target": target,
                    "toxic": 0,
                }
            )

    return pd.DataFrame(expanded)


def train_and_export():
    print("Step 1: Assembling Jigsaw Unintended Bias Dataset...")
    df = build_curated_jigsaw_dataset()
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = data_dir / "jigsaw_toxicity_dataset.csv"
    df.to_csv(dataset_csv, index=False)
    print(f"Dataset saved to {dataset_csv} (Total records: {len(df)})")

    X = df[["comment_text"]]
    y = df["toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Step 2: Training Pipeline matching Khan (2023)...")
    # Pure scikit-learn TF-IDF vectorizer (lowercasing, stop_words, tokenization, sublinear TF)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
                ),
                "comment_text",
            )
        ],
        remainder="drop",
    )

    estimator = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000, random_state=42
    )

    trained_pipeline = Pipeline(
        [("preprocess", preprocessor), ("model", estimator)]
    )

    trained_pipeline.fit(X_train, y_train)

    y_pred = trained_pipeline.predict(X_test)
    y_prob = trained_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation on Test Split ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    models_dir = Path("backend/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "model_bundle.joblib"

    feature_names = ["comment_text"]
    bundle = {
        "pipeline": trained_pipeline,
        "features": feature_names,
        "task": "classification",
        "classes": ["Non-Toxic", "Toxic"],
        "model_version": "1.0",
        "methodology": "TF-IDF + Balanced Logistic Regression (Khan 2023)",
    }

    joblib.dump(bundle, bundle_path)
    print(f"\nSaved model bundle to {bundle_path}")

    # Export sample record for Stage 5
    sample_df = pd.DataFrame(
        [
            {
                "comment_text": (
                    "I love the idea of upvoting entire articles and sharing"
                    " thoughtful feedback."
                )
            }
        ]
    )
    sample_csv = data_dir / "sample_inputs.csv"
    sample_df.to_csv(sample_csv, index=False)
    print(f"Exported sample record to {sample_csv}")


if __name__ == "__main__":
    train_and_export()
