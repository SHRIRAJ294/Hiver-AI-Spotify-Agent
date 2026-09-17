import pandas as pd

file = "data/new_intent_examples.csv"

df = pd.read_csv(file)

# Move clear playback problems to playback_audio
playback_messages = [
    "Desktop app isn't playing songs, but it's playing on my phone. I have restarted and uninstalled the app with no joy",
    "Looks like you're down again today, won't play on my desktop in Chrome or Firefox.",
    "notifications still stop my music so i have to force quit the app and play the song back over.",
    "A day later, computer turned off overnight, and it's still frozen on the same song. The player isn't responding to any clicks."
]

df.loc[
    df["customer_message"].isin(playback_messages),
    "intent"
] = "playback_audio"


# Remove examples that contain only context and no actual issue
remove_messages = [
    "iPhone 6s with the most current operating system and Spotify version with a premium family account",
    "The past 2 days. Laptop. Spotify Premium. Newest desktop update."
]

df = df[
    ~df["customer_message"].isin(remove_messages)
]

df.to_csv(file, index=False)

print("Training examples cleaned.")
print(df["intent"].value_counts())