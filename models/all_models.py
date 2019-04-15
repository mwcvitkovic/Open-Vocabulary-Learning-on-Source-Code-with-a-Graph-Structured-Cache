# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from models.FITB.CharCNN import FITBCharCNN
from models.FITB.ClosedVocab import FITBClosedVocab
from models.FITB.GSCVocab import FITBGSCVocab
from models.GraphNN.DTNN import DTNN
from models.GraphNN.GAT import GAT
from models.GraphNN.GGNN import GGNN
from models.GraphNN.RGCN import RGCN
from models.VarNaming.CharCNN import VarNamingCharCNN
from models.VarNaming.ClosedVocab import VarNamingClosedVocab
from models.VarNaming.GSCVocab import VarNamingGSCVocab


class FITBClosedVocabGGNN(FITBClosedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNGGNN(FITBCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBGSCVocabGGNN(FITBGSCVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingClosedVocabGGNN(VarNamingClosedVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNGGNN(VarNamingCharCNN, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingGSCVocabGGNN(VarNamingGSCVocab, GGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBClosedVocabDTNN(FITBClosedVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNDTNN(FITBCharCNN, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBGSCVocabDTNN(FITBGSCVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingClosedVocabDTNN(VarNamingClosedVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNDTNN(VarNamingCharCNN, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingGSCVocabDTNN(VarNamingGSCVocab, DTNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBClosedVocabRGCN(FITBClosedVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNRGCN(FITBCharCNN, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBGSCVocabRGCN(FITBGSCVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingClosedVocabRGCN(VarNamingClosedVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNRGCN(VarNamingCharCNN, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingGSCVocabRGCN(VarNamingGSCVocab, RGCN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBClosedVocabGAT(FITBClosedVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBCharCNNGAT(FITBCharCNN, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FITBGSCVocabGAT(FITBGSCVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingClosedVocabGAT(VarNamingClosedVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingCharCNNGAT(VarNamingCharCNN, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VarNamingGSCVocabGAT(VarNamingGSCVocab, GAT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
