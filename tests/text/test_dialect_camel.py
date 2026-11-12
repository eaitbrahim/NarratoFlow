import importlib.util

import pytest

from narratoflow.errors import DialectUnavailableError
from narratoflow.text.dialect import load_camel_model


def test_real_model_tells_msa_from_egyptian() -> None:
    if importlib.util.find_spec("camel_tools") is None:
        pytest.skip("camel_tools is not installed")
    try:
        model = load_camel_model()
    except DialectUnavailableError:
        pytest.skip("dialectid-model26 data is not installed")
    msa = "\u0630\u0647\u0628 \u0627\u0644\u0637\u0627\u0644\u0628 "
    msa += "\u0625\u0644\u0649 \u0627\u0644\u0645\u062f\u0631\u0633\u0629"
    egy = "\u0625\u0632\u064a\u0643 \u0639\u0627\u0645\u0644 \u0625\u064a\u0647 "
    egy += "\u0627\u0644\u0646\u0647\u0627\u0631\u062f\u0647"
    (msa_label, _), (egy_label, _) = model.predict([msa, egy])
    assert msa_label == "MSA"
    assert egy_label != "MSA"
