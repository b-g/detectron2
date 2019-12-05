## Install
1. `nvidia-docker build -t detectron2:v0 .`
2. `nvidia-docker run -it --name detectron2 detectron2:v0`

Do use the repo from the outside via a docker volume, you have to build the repo once
3. python3 setup.py build develop

## Running
./run-docker.sh

inside docker:
python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input output/input.jpg --output output/output.png \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

python3 demo/mask.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input-folder output/test/frames --output-folder output/test/masked \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
