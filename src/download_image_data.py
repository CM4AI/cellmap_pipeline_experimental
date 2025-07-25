import json

from cellmaps_imagedownloader.gene import ImageGeneNodeAttributeGenerator
from cellmaps_imagedownloader.proteinatlas import (ImageDownloadTupleGenerator,
                                                   ProteinAtlasProcessor,
                                                   ProteinAtlasImageUrlReader,
                                                   ProteinAtlasReader)
from cellmaps_imagedownloader.runner import (CellmapsImageDownloader,
                                             MultiProcessImageDownloader)

import mlflow
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("image_downloader")
ml_logger = LoggerFactory.get_logger("mlflow")

with mlflow.start_run() as run:
    mlflow.set_tag("pipeline_step", "cellmaps_image_downloader")

    outdir = f"data/images/{run.info.run_id}"

    json_prov = None
    with open("examples/provenance.json", 'r') as f:
        json_prov = json.load(f)

    ## Download by cell line
    cell_line = "U2OS"
    mlflow.log_param("image_source", "hpa")
    mlflow.log_param("cell_line", cell_line)
    
    created_outdir = True
    hpa_processor = ProteinAtlasProcessor(
        outdir,
        ProteinAtlasReader.DEFAULT_PROTEINATLAS_URL,
        None,
        cell_line
    )
    samples, proteinatlasxml = hpa_processor.get_sample_list_from_hpa()
    samples_list = ImageGeneNodeAttributeGenerator.get_samples_from_csvfile(samples)
    unique_list = None

    ## Download by sample list
    # created_outdir = False
    # mlflow.log_param("image_source", "cellmaps_pipeline_example")
    # samples_list = ImageGeneNodeAttributeGenerator.get_samples_from_csvfile("examples/samples.csv")
    # unique_list = ImageGeneNodeAttributeGenerator.get_unique_list_from_csvfile("examples/unique.csv")

    ## Download images
    dloader = MultiProcessImageDownloader(
        poolsize=24,
        skip_existing=True
    )

    imagegen = ImageGeneNodeAttributeGenerator(
        unique_list=unique_list,
        samples_list=samples_list
    )

    proteinatlas_reader = ProteinAtlasReader(outdir, proteinatlas=ProteinAtlasReader.DEFAULT_PROTEINATLAS_URL)
    proteinatlas_urlreader = ProteinAtlasImageUrlReader(reader=proteinatlas_reader)

    imageurlgen = ImageDownloadTupleGenerator(
        reader=proteinatlas_urlreader,
        samples_list=imagegen.get_samples_list(),
        valid_image_ids=imagegen.get_samples_list_image_ids()
    )

    img_downloader = CellmapsImageDownloader(
        outdir=outdir,
        imagedownloader=dloader,
        imgsuffix=CellmapsImageDownloader.IMG_SUFFIX,
        imagegen=imagegen,
        imageurlgen=imageurlgen,
        skip_logging=True,
        provenance=json_prov,
        skip_failed=True,
        existing_outdir=created_outdir,
        log_fairops=True
    )

    img_downloader.run()

    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
