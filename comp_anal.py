import pandas as pd
from matplotlib import pyplot as plt

apple_df = pd.read_csv("apple_SA.csv")
samsung_df = pd.read_csv("samsung_SA.csv")

apple_sentiments = apple_df.groupby("sentiment_label")["comment_text"].count()
samsung_sentiments = samsung_df.groupby("sentiment_label")["comment_text"].count()

fig, ax = plt.subplots()
apple_sentiments.plot(kind="bar", ax=ax, position=0, width=0.4, label="Apple", color="Orange")
samsung_sentiments.plot(kind="bar", ax=ax, position=1, width=0.4, label="Samsung", color="Blue")

ax.set_xlabel("Type of Sentiment")
ax.set_ylabel("Number of Tweets")
ax.set_title("Apple v. Samsung")
plt.show()