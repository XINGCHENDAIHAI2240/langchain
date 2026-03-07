import os

import dotenv

dotenv.load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
TONGYI_BASE_URL = os.getenv("TONGYI_BASE_URL")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")
