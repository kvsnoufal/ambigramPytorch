txts=[
    "yeah","BUNNY","ambigram","renan",'chump',"dubai",'emirates',
    "honey",'OHIO','ohio','CLOUD','cANDY','hayah','HayaH',
    'python','paul','doug','dog',"LiFT",
    "coffee","big",'fizzy',"Ambiguity","raHul","Fire","MARIAM"
]
from glob import glob
fonts = glob("fonts/*.ttf")

iterations = 100

output_path="output/init"
write_path="results/refine_svg"

device="cuda"

embeddingLossWeight=0.001