import json

from cellmaps_ppidownloader.gene import (APMSGeneNodeAttributeGenerator,
                                         NdexGeneNodeAttributeGenerator)
from cellmaps_ppidownloader.runner import CellmapsPPIDownloader
import mlflow
from fairops.mlops.autolog import LoggerFactory

mlflow.set_experiment("ppi_downloader")
ml_logger = LoggerFactory.get_logger("mlflow")

with mlflow.start_run() as run:
    mlflow.set_tag("pipeline_step", "cellmaps_ppi_downloader")

    outdir = f"data/ppi/{run.info.run_id}"

    json_prov = None
    with open("examples/provenance.json", 'r') as f:
        json_prov = json.load(f)

    # apmsgen = APMSGeneNodeAttributeGenerator(
    #     apms_edgelist=APMSGeneNodeAttributeGenerator.get_apms_edgelist_from_tsvfile("examples/edgelist.tsv",
    #                                                                                 geneid_one_col=APMSGeneNodeAttributeGenerator.GENEID_COL1,
    #                                                                                 symbol_one_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL1,
    #                                                                                 geneid_two_col=APMSGeneNodeAttributeGenerator.GENEID_COL2,
    #                                                                                 symbol_two_col=APMSGeneNodeAttributeGenerator.SYMBOL_COL2),
    #     apms_baitlist=APMSGeneNodeAttributeGenerator.get_apms_baitlist_from_tsvfile("examples/baitlist.tsv",
    #                                                                                 symbol_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_SYMBOL,
    #                                                                                 geneid_col=APMSGeneNodeAttributeGenerator.BAITLIST_GENE_ID,
    #                                                                                 numinteractors_col=APMSGeneNodeAttributeGenerator.BAITLIST_NUM_INTERACTORS))

    UUID = "95bc75d5-d1d1-11ee-8a40-005056ae23aa"
    mlflow.log_param("ppi_source", "ndex")
    mlflow.log_param("ndex_uuid", UUID)

    ndexgen = NdexGeneNodeAttributeGenerator(uuid=UUID,
                                            apms_edgelist=NdexGeneNodeAttributeGenerator.get_apms_edgelist_from_ndex(uuid=UUID),
                                            apms_baitlist=NdexGeneNodeAttributeGenerator.get_apms_baitlist_from_ndex(uuid=UUID)
    )

    ppi_downloader = CellmapsPPIDownloader(
        outdir=outdir,
        apmsgen=ndexgen,
        skip_logging=True,
        provenance=json_prov)

    ppi_downloader.run()

    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
