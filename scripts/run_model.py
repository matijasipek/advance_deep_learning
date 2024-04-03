import argparse, json, os
from imageio import imwrite
import torch
import sys
import os

project_root = '/zhome/48/2/181238/cv_project/advance_deep_learning'
sys.path.append(project_root)

from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='pretrained_models/vg64.pt')
parser.add_argument('--scene_graphs_json', default='data/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=1)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  # Load the scene graphs
  with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)

  # Run the model forward
  with torch.no_grad():
    imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
  imgs = imagenet_deprocess_batch(imgs)

  # Save the generated images
  for i in range(imgs.shape[0]):
    img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)

  # Draw the scene graphs
  if args.draw_scene_graphs == 1:
    for i, sg in enumerate(scene_graphs):
      sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
      sg_img_path = os.path.join(args.output_dir, 'sg%03d.png' % i)
      imwrite(sg_img_path, sg_img)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)