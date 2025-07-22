import json

from cellmaps_ppidownloader.gene import (APMSGeneNodeAttributeGenerator,
                                         NdexGeneNodeAttributeGenerator)
from cellmaps_ppidownloader.runner import CellmapsPPIDownloader


json_prov = None
with open("examples/provenance.json", 'r') as f:
    json_prov = json.load(f)

#apmsgen = APMSGeneNodeAttributeGenerator(
#    apms_edgelist=APMSGeneNodeAttributeGenerator.get_apms_edgelist_from_tsvfile("examples/edgelist.tsv",
#                                                                                geneid_one_col=APMSGeneNodeAttributeGenerator.GENEID_COL1,
#                                                                                symbol_one_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL1,
#                                                                                geneid_two_col=APMSGeneNodeAttributeGenerator.GENEID_COL2,
#                                                                                symbol_two_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL2),
#    apms_baitlist=APMSGeneNodeAttributeGenerator.get_apms_baitlist_from_tsvfile("examples/baitlist.tsv",
#                                                                                symbol_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_SYMBOL,
#                                                                                geneid_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_ID,
#                                                                                numinteractors_col=APMSGeneNodeAttributeGenerator.BAITLIST_NUM_INTERACTORS))

UUID = "95bc75d5-d1d1-11ee-8a40-005056ae23aa"

ndexgen = NdexGeneNodeAttributeGenerator(uuid=UUID,
                                         apms_edgelist=NdexGeneNodeAttributeGenerator.get_apms_edgelist(uuid=UUID),
                                         apms_baitlist=NdexGeneNodeAttributeGenerator.get_apms_baitlist_from_ndex(uuid=UUID)
)

ppi_downloader = CellmapsPPIDownloader(
    outdir="data/ppi",
    apmsgen=ndexgen,
    skip_logging=True,
    provenance=json_prov)

ppi_downloader.run()