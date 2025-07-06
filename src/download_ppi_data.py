import json

from cellmaps_ppidownloader.runner import CellmapsPPIDownloader
from cellmaps_ppidownloader.gene import APMSGeneNodeAttributeGenerator
from cellmaps_ppidownloader.gene import CM4AIGeneNodeAttributeGenerator


json_prov = None
with open("examples/provenance.json", 'r') as f:
    json_prov = json.load(f)

apmsgen = APMSGeneNodeAttributeGenerator(
    apms_edgelist=APMSGeneNodeAttributeGenerator.get_apms_edgelist_from_tsvfile("examples/edgelist.tsv",
                                                                                geneid_one_col=APMSGeneNodeAttributeGenerator.GENEID_COL1,
                                                                                symbol_one_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL1,
                                                                                geneid_two_col=APMSGeneNodeAttributeGenerator.GENEID_COL2,
                                                                                symbol_two_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL2),
    apms_baitlist=APMSGeneNodeAttributeGenerator.get_apms_baitlist_from_tsvfile("examples/baitlist.tsv",
                                                                                symbol_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_SYMBOL,
                                                                                geneid_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_ID,
                                                                                numinteractors_col=APMSGeneNodeAttributeGenerator.BAITLIST_NUM_INTERACTORS))

ppi_downloader = CellmapsPPIDownloader(
    outdir="data/ppi",
    apmsgen=apmsgen,
    skip_logging=True,
    provenance=json_prov)

ppi_downloader.run()