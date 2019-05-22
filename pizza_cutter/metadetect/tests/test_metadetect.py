try:
    from metadetect.test import test_metadetect as test_func
except Exception:
    from metadetect.test_metadetect import test_metadetect as test_func


def test_metadetect():
    """
    run the test from metadetect
    """
    test_func()
