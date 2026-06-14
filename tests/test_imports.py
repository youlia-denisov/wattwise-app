import unittest


class TestImports(unittest.TestCase):
    """
    Verify every src module and the pipeline can be imported without error.
    A failure here means a missing import or undefined name at module level.
    Run these first — if they fail, nothing else will work.
    """

    def test_import_config(self):
        import config

    def test_import_clustering(self):
        import src.clustering

    def test_import_visualization(self):
        import src.visualization

    def test_import_features(self):
        import src.features

    def test_import_preprocessing(self):
        import src.preprocessing

    def test_import_aggregation(self):
        import src.aggregation

    def test_import_outliers(self):
        import src.outliers

    def test_import_discount_analysis(self):
        import src.discount_analysis

    def test_import_discount_calculator(self):
        import src.discount_calculator

    def test_import_loader(self):
        import src.loader

    def test_import_pipeline(self):
        import pipeline


if __name__ == "__main__":
    unittest.main(verbosity=2)
