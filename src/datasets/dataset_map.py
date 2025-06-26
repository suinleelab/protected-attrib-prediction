from src.datasets.cxrrace import CheXpertRaceDataset, MIMICCXRRaceDataset
from src.datasets.engineered import CheXpertDiagnosisDataset, MIMICCXRDiagnosisDataset, MelanomaDataset
from src.datasets.isicsex import ISICSexDataset
from src.datasets.isicbalanced import BalancedBinaryHairISICSexDataset

dataset_map = {
        'CheXpertRaceDataset': CheXpertRaceDataset,
        'MIMICCXRRaceDataset': MIMICCXRRaceDataset,
        'CheXpertDiagnosisDataset': CheXpertDiagnosisDataset,
        'MIMICCXRDiagnosisDataset': MIMICCXRDiagnosisDataset,
        'ISICSexDataset': ISICSexDataset,
        'BalancedBinaryHairISICSexDataset': BalancedBinaryHairISICSexDataset,
        'MelanomaDataset': MelanomaDataset
}