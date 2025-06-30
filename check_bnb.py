import torch, torchvision, torchaudio
print(torch.__version__, "CUDA:", torch.version.cuda)
print(torchvision.__version__)
print(torchaudio.__version__)

from langchain_community.vectorstores import Chroma
Chroma.__module__
'langchain.vectorstores.chroma'