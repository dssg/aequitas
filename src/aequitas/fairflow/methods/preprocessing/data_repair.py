from aif360.algorithms.preprocessing import DisparateImpactRemover

from ..preprocessing import PreProcessing


class DataRepair(PreProcessing):
    def __init__(self, ):
        self.repair = DisparateImpactRemover()

    def fit():
        pass
