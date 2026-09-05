import kagglehub
print("Downloading dataset...")
path = kagglehub.competition_download('home-credit-default-risk')
print(f"Downloaded path: {path}")
