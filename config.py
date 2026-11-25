"""Shared settings for the Kenyan Farmer Query Routing project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models" / "mbert-farmer-finetuned"

MODEL_NAME = "bert-base-multilingual-cased"   # mBERT (allowed by the brief)
LABELS = ["pests", "crop_disease", "soil_fertilizer", "weather_climate", "market_prices", "livestock"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SEED = 42
MAX_LEN = 48
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Fine-tuning hyperparameters
EPOCHS = 10
LR = 3e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

# Label descriptions used by the zero-shot base model (English + Swahili keywords)
LABEL_DESCRIPTIONS = {
    "pests": "insect pests eating crops, armyworm, aphids, weevils, rats. wadudu, viwavi, panya",
    "crop_disease": "plant disease, blight, rot, fungus, virus, leaf spots, wilting. ugonjwa wa mimea, madoa, kuoza",
    "soil_fertilizer": "soil fertility, fertilizer, manure, compost, lime, DAP. udongo, mbolea, samadi, mboji",
    "weather_climate": "weather, rain, drought, floods, forecast, irrigation. hali ya hewa, mvua, ukame, mafuriko",
    "market_prices": "market price, selling produce, buyers, brokers, cooperative. bei, soko, kuuza, mnunuzi",
    "livestock": "livestock, cattle, goats, chickens, milk, animal disease, vaccine. mifugo, ng'ombe, mbuzi, kuku",
}
