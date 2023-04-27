from matplotlib import pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# read csv
df = pd.read_csv("twitter-sentiment-analysis.csv")

# initialize sia object
sia = SentimentIntensityAnalyzer()

# compile sentiment score for each tweet
df["scores"] = df["tweet"].apply(lambda tweet: sia.polarity_scores(str(tweet)))

df["compound"] = df["scores"].apply(lambda scores_dict: scores_dict["compound"])
df["positive"] = df["scores"].apply(lambda scores_dict: scores_dict["pos"])
df["negative"] = df["scores"].apply(lambda scores_dict: scores_dict["neg"])
df["neutral"] = df["scores"].apply(lambda scores_dict: scores_dict["neu"])

# Classify each tweet based on Compound Score
df["type"] = ""
df.loc[df.compound > 0, "type"] = "POSITIVE"
df.loc[df.compound < 0, "type"] = "NEGATIVE"
df.loc[df.compound == 0, "type"] = "NEUTRAL"


# Count total values
len = df.shape
(rows, cols) = len

# Count Pos, Neg & Neu values
POSITIVE = 0
NEGATIVE = 0
NEUTRAL = 0
for i in range(0, rows):
    if df.loc[i][8] == "POSITIVE":
        POSITIVE+=1
    if df.loc[i][8] == "NEGATIVE":
        NEGATIVE+=1
    if df.loc[i][8] == "NEUTRAL":
        NEUTRAL+=1
print(f"Positive: {POSITIVE}, Negative: {NEGATIVE}, Neutral: {NEUTRAL}")

# Visualization
fig = plt.figure(figsize=(10, 5))
plt.bar(["Positive", "Negative", "Neutral"], [POSITIVE, NEGATIVE, NEUTRAL], width=0.4)
plt.xlabel("Type of Sentiment")
plt.ylabel("Sentiment Score")
plt.title("Tweet Analysis")
plt.show()
