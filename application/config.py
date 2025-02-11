import os
from dotenv import load_dotenv

# Change CWD to parent directory of 'application' if needed
current_dir = os.getcwd()

# Move one level up if inside "application" directory
if os.path.basename(current_dir) == "application":
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    print(f"🔄 Changing working directory to: {parent_dir}")
    os.chdir(parent_dir)


# Print updated working directory
print(f"📁 Current Working Directory: {os.getcwd()}")

# Load the correct .env file based on FLASK_ENV
env = os.getenv("FLASK_ENV", "development")
dotenv_file = f".env.{env}"

# Check if .env file exists before loading
if os.path.exists(dotenv_file):
    print(f"✅ Loading environment variables from: {dotenv_file}")
    load_dotenv(dotenv_file)
else:
    print(f"⚠️ Warning: {dotenv_file} not found! Using system environment variables.")

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    OPEN_API_KEY = os.getenv("OPEN_API_KEY")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    REGION_NAME = os.getenv("REGION_NAME")
    S3_BUCKET = os.getenv("S3_BUCKET")
    TECH_INDEX_DIR = os.getenv("TECH_INDEX_DIR")

# Debugging prints
print(f"TECH_INDEX_DIR Loaded: {Config.TECH_INDEX_DIR}")  # Should print correct path or None
