from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def pair_frame() -> pd.DataFrame:
    rows = []
    topics = ["python", "guitar", "algebra", "photography", "coffee", "running"]
    for index in range(80):
        topic = topics[index % len(topics)]
        if index % 2 == 0:
            q1 = f"How can I learn {topic} lesson {index}?"
            q2 = f"What is the best way to learn {topic} lesson {index}?"
            label = 1
        else:
            q1 = f"How can I learn {topic} lesson {index}?"
            q2 = f"Where can I buy a bicycle model {index}?"
            label = 0
        rows.append(
            {
                "id": index,
                "question1": q1,
                "question2": q2,
                "is_duplicate": label,
            }
        )
    return pd.DataFrame(rows)
