import pytest

import matchcake_opt


@pytest.mark.parametrize(
    "attr",
    [
        "__author__",
        "__email__",
        "__copyright__",
        "__license__",
        "__url__",
        "__package__",
        "__version__",
    ],
)
def test_attributes(attr):
    assert hasattr(matchcake_opt, attr), f"Module does not have attribute {attr}"
    assert getattr(matchcake_opt, attr) is not None, f"Attribute {attr} is None"
    assert isinstance(getattr(matchcake_opt, attr), str), f"Attribute {attr} is not a string"
