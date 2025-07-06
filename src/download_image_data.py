import json

from cellmaps_imagedownloader.gene import ImageGeneNodeAttributeGenerator
from cellmaps_imagedownloader.proteinatlas import (ImageDownloadTupleGenerator,
                                                   ProteinAtlasImageUrlReader,
                                                   ProteinAtlasReader)
from cellmaps_imagedownloader.runner import (CellmapsImageDownloader,
                                             MultiProcessImageDownloader)

json_prov = None
with open("examples/provenance.json", 'r') as f:
    json_prov = json.load(f)


samples_list = ImageGeneNodeAttributeGenerator.get_samples_from_csvfile("examples/samples.csv")
unique_list = ImageGeneNodeAttributeGenerator.get_unique_list_from_csvfile("examples/unique.csv")

dloader = MultiProcessImageDownloader(
    poolsize=MultiProcessImageDownloader.POOL_SIZE,
    skip_existing=True
)

imagegen = ImageGeneNodeAttributeGenerator(
    unique_list=unique_list,
    samples_list=samples_list
)

proteinatlas_reader = ProteinAtlasReader("data/images", proteinatlas=ProteinAtlasReader.DEFAULT_PROTEINATLAS_URL)
proteinatlas_urlreader = ProteinAtlasImageUrlReader(reader=proteinatlas_reader)

imageurlgen = ImageDownloadTupleGenerator(
    reader=proteinatlas_urlreader,
    samples_list=imagegen.get_samples_list(),
    valid_image_ids=imagegen.get_samples_list_image_ids()
)

img_downloader = CellmapsImageDownloader(
    outdir="data/images",
    imagedownloader=dloader,
    imgsuffix=CellmapsImageDownloader.IMG_SUFFIX,
    imagegen=imagegen,
    imageurlgen=imageurlgen,
    skip_logging=True,
    provenance=json_prov,
    skip_failed=True,
    existing_outdir=False
)

img_downloader.run()