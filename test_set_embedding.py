import time
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

df = pd.read_parquet(r"data\ATML2024_reviews_test.parquet")

text_list = df["text"].tolist()


def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "CLASSIFICATION",
    model_name: str = "text-embedding-004",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]


text_list_embeddings_256 = []


def batch_data(data, batch_size=80):
    """Yield successive batch-sized chunks from data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


data_batches = list(batch_data(text_list))
for batch in tqdm(data_batches):
    while True:
        try:
            text_list_embeddings_256.extend(embed_text(texts=batch, dimensionality=256))
            break
        except:
            time.sleep(5)

pd.Series(text_list_embeddings_256).to_pickle("data/embeddings_submission.pkl")
