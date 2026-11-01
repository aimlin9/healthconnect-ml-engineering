import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from hc_pipeline import config, load
from hc_pipeline.features import add_prior_appointments


@pytest.fixture(scope="session")
def features_df():
    return add_prior_appointments(pd.read_csv(config.FEATURES_CSV))


@pytest.fixture(scope="session")
def split(features_df):
    y = features_df[config.TARGET]
    tr, te = train_test_split(features_df.index, test_size=config.TEST_SIZE,
                              random_state=config.RANDOM_STATE, stratify=y)
    return tr, te


@pytest.fixture(scope="session")
def artifact():
    return load()


@pytest.fixture
def sample_rows(features_df, split):
    """Ten held-out rows with every category represented, as a client would send them."""
    _, te = split
    return features_df.loc[te, config.FEATURES].head(10).copy()
