mkdir checkpoint
wget -O bestmodel_PbA.pth 'https://www.dropbox.com/scl/fi/hmgfvwgs9zjcyit0evuqp/model_epoch146.pth?rlkey=0zlfapa99dj4ijs00ej2dg1j0&st=cehg69su&dl=1'
mv bestmodel_PbA.pth checkpoint/
wget -O bestmodel_PbB.pth 'https://www.dropbox.com/scl/fi/q715wy15i386m9w47nox4/modelB_585.pth?rlkey=eknifj4c3jkqqznryw3rd597n&st=8undrvei&dl=1'
mv bestmodel_PbB.pth checkpoint/
python3 bestmodel_PbB_download.py