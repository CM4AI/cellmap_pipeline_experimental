import json

from cellmaps_imagedownloader.gene import ImageGeneNodeAttributeGenerator
from cellmaps_imagedownloader.proteinatlas import (ImageDownloadTupleGenerator,
                                                   ProteinAtlasProcessor,
                                                   ProteinAtlasImageUrlReader,
                                                   ProteinAtlasReader)
from cellmaps_imagedownloader.runner import (CellmapsImageDownloader,
                                             MultiProcessImageDownloader)


outdir = "data/images-u2os"

json_prov = None
with open("examples/provenance.json", 'r') as f:
    json_prov = json.load(f)

## Download by cell line
cell_line = "U2OS"
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
# samples_list = ImageGeneNodeAttributeGenerator.get_samples_from_csvfile("examples/samples.csv")
# unique_list = ImageGeneNodeAttributeGenerator.get_unique_list_from_csvfile("examples/unique.csv")

## Download images
dloader = MultiProcessImageDownloader(
    poolsize=MultiProcessImageDownloader.POOL_SIZE,
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
    existing_outdir=created_outdir
)

img_downloader.run()